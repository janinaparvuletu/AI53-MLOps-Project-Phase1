
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import torch


def training_data_identity(
    train_manifest: Path = Path("data/processed/manifests/train.csv"),
    validation_manifest: Path = Path("data/processed/manifests/validation.csv"),
) -> str:
    """Return a deterministic identity for training and validation data."""
    paths = (train_manifest, validation_manifest)

    if any(not path.exists() for path in paths):
        return "unavailable"

    digest = hashlib.sha256()

    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return digest.hexdigest()


def get_package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unavailable"


def get_git_info() -> dict:
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )

    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )

    return {
        "commit_sha": (
            commit_result.stdout.strip()
            if commit_result.returncode == 0
            else "unavailable"
        ),
        "dirty": (
            bool(status_result.stdout.strip())
            if status_result.returncode == 0
            else None
        ),
    }


def build_run_manifest(params: dict, device: str, mlflow_run_id: str) -> dict:
    """Build experiment provenance metadata."""

    train = params["train"]
    data = params["data"]
    augmentation = params["augmentation"]

    return {
        "git": get_git_info(),
        "data": {
            "training_data_identity": training_data_identity(),
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torchvision": get_package_version("torchvision"),
            "dvc": get_package_version("dvc"),
            "mlflow": get_package_version("mlflow"),
            "cuda_runtime": torch.version.cuda or "unavailable",
        },
        "configuration": {
            "image_size": data["image_size"],
            "batch_size": train["batch_size"],
            "epochs": train["epochs"],
            "learning_rate": train["learning_rate"],
            "dropout": train["dropout"],
            "augmentation_enabled": augmentation["enabled"],
            "horizontal_flip": augmentation["horizontal_flip"],
            "rotation_degrees": augmentation["rotation_degrees"],
        },
        "settings": {
            "device": device,
            "seed": train["random_state"],
        },
        "execution": {
            "mlflow_run_id": mlflow_run_id,
            "random_state": train["random_state"],
        },
    }


def write_run_manifest(
    params: dict,
    device: str,
    mlflow_run_id: str,
    path: Path,
) -> dict:
    manifest = build_run_manifest(params, device, mlflow_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest
