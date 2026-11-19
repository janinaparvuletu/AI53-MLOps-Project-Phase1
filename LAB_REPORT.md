# Milestone 0 – Starter Audit

| File / Setting | Problem identified | Proposed fix |
|---|---|---|
| src/reproducibility.py | NumPy random generator is not seeded. | Implement np.random.seed(). |
| src/reproducibility.py | PyTorch CPU and CUDA generators are not seeded. | Seed PyTorch CPU and CUDA generators. |
| src/reproducibility.py | cuDNN deterministic settings are missing. | Enable deterministic mode and disable benchmark mode. |
| src/reproducibility.py | DataLoader workers are not seeded. | Implement seed_worker() for Python and NumPy. |
| src/reproducibility.py | DataLoader generator is not seeded. | Implement make_generator() using the provided seed. |
| src/train.py | Run provenance manifest is not captured before training. | Call write_run_manifest() at the beginning of the MLflow run. |
| src/train.py | Training parameters and data identity are not logged. | Log configuration parameters and training data identity to MLflow. |
| src/train.py | Training dataset is not registered as an MLflow input. | Create and log an MLflow Dataset from the training manifest. |
| src/train.py | Training and validation metrics are not logged per epoch. | Log loss, accuracy and macro-F1 metrics with epoch steps. |
| src/train.py | Training duration is not logged. | Log training_seconds to MLflow. |
| src/train.py | Training artifacts are not logged to MLflow. | Log curves, summary and provenance manifest. |
| src/train.py | Final PyTorch model is not logged to the MLflow Model Registry. | Log the model with signature, input example and metadata. |

