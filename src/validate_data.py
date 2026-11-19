
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

EXPECTED_SPLITS = {"train": 1034, "validation": 133, "test": 128}
EXPECTED_CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_dataset(raw_root: Path) -> dict:
    report = {
        "status": "PASS",
        "total_images": sum(EXPECTED_SPLITS.values()),
        "splits": {},
    }

    for split, expected_count in EXPECTED_SPLITS.items():
        split_dir = raw_root / split

        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split: {split}")

        present_classes = {
            p.name for p in split_dir.iterdir() if p.is_dir()
        }

        if present_classes != EXPECTED_CLASSES:
            raise RuntimeError(
                f"Unexpected classes in {split}: {sorted(present_classes)}"
            )

        class_report = {}
        total_images = 0

        for class_name in sorted(EXPECTED_CLASSES):
            class_dir = split_dir / class_name

            images = sorted(
                p for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )

            if not images:
                raise RuntimeError(
                    f"Empty class: {split}/{class_name}"
                )

            for image_path in images:
                with Image.open(image_path) as image:
                    image.load()

            class_report[class_name] = len(images)
            total_images += len(images)

        if total_images != expected_count:
            raise RuntimeError(
                f"Unexpected image count in {split}: "
                f"{total_images}; expected {expected_count}"
            )

        report["splits"][split] = {
            "total_images": total_images,
            "classes": class_report,
        }

    return report


def main() -> None:
    report = validate_dataset(Path("data/raw"))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
