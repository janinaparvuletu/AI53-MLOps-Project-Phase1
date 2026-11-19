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


## Milestone 6 – DVC recomputation predictions

| Change | VALIDATE | PREPARE | TRAIN | EVALUATE |
|---|---|---|---|---|
| Change train.learning_rate | REUSED | REUSED | RERUN | REUSED |
| Change data.image_size | REUSED | RERUN | RERUN | NEEDS UPDATE |
| Edit src/validate_data.py; quality report stays the same | RERUN | REUSED | REUSED | REUSED |
| Edit only README.md | REUSED | REUSED | REUSED | REUSED |

### Engineering checkpoint 5

Changing train.learning_rate reruns only TRAIN because it is the only stage that declares the train parameters as dependencies. EVALUATE keeps the same selected Run ID, so its inputs do not change.

Changing data.image_size reruns PREPARE because it declares data.image_size and reads data/raw. TRAIN must rerun because it also declares data.image_size and uses the prepared training data. EVALUATE needs updated results because it declares data.image_size and uses the prepared test data.

Editing src/validate_data.py reruns VALIDATE because the file is one of its dependencies. Even if the quality report remains identical, PREPARE does not depend on that report, so the other stages can reuse their outputs.

### Milestone 6 – Model selection criteria

- Primary validation metric: Macro-F1.
- Tie-break rule: choose the Run with higher validation accuracy if Macro-F1 is equal.
- Learning-rate prediction: reducing the learning rate from 0.001 to 0.0003 will produce smaller parameter updates and potentially slower, more gradual learning. With only four epochs, it may also result in slower convergence.


### Run B – Lower learning rate

- Learning rate: 0.0003
- Run ID: bf67f7864ffc4b428123a8e15d5699d3
- Final validation Macro-F1: 0.6091
- Final validation Accuracy: 0.6165
- Final validation Loss: 0.8927

Compared with Run A (learning rate 0.001), Run B achieved lower validation Macro-F1 and accuracy, and higher validation loss. Therefore, reducing the learning rate to 0.0003 did not improve performance within four epochs.

#### DVC recomputation – Run B

| Change | VALIDATE | PREPARE | TRAIN | EVALUATE |
|---|---|---|---|---|
| Change train.learning_rate from 0.001 to 0.0003 | REUSED | REUSED | RERUN | NOT RUN |

### Run C – Data augmentation

- Learning rate: 0.001
- Augmentation: enabled
- Prediction: enabling data augmentation may improve validation Macro-F1 by helping the model generalize better to image variations.
- Only the augmentation setting is changed compared with Run A.
