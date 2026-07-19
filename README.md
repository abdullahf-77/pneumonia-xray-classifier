# Pneumonia Detection — Chest X-Ray Classifier

A transfer-learning image classifier that detects pneumonia from chest X-ray
images, built with TensorFlow/Keras and MobileNetV2. Developed as a deep
learning coursework/exhibition project.

## Overview

The notebook downloads the public
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
dataset from Kaggle, fine-tunes a MobileNetV2 backbone for binary
classification (`NORMAL` vs. `PNEUMONIA`), and evaluates the result with
accuracy/loss curves and a confusion matrix / classification report.

## Features

- Automatic dataset download via `kagglehub` (no manual upload needed)
- Transfer learning on `MobileNetV2` (ImageNet weights, frozen backbone)
- Binary image classification with a lightweight classification head
  (`GlobalAveragePooling2D` → `Dense(64, relu)` → `Dropout(0.3)` → `Dense(1, sigmoid)`)
- Training/validation accuracy and loss curves
- Confusion matrix and classification report on the held-out test set
- Single-image inference cell for manual upload/testing (Colab)

## Tech Stack

| Category        | Technology                     |
|------------------|---------------------------------|
| Language          | Python                         |
| Deep Learning     | TensorFlow / Keras (MobileNetV2) |
| Data              | kagglehub, Kaggle Chest X-Ray Pneumonia dataset |
| Evaluation        | scikit-learn (confusion matrix, classification report) |
| Visualization     | Matplotlib                     |
| Environment       | Google Colab / Jupyter Notebook |

## Folder Structure

```
Pneumonia_Detection/
├── ChestX-RayProject_123.ipynb   # Main notebook: data, training, evaluation
├── requirements.txt              # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

> Note: the dataset itself (`chest_xray/`) is **not** included in this
> repository. It is downloaded automatically at runtime via `kagglehub`.

## Requirements

- Python 3.10+
- A Kaggle account is **not** required to run the notebook in Colab (the
  dataset is served through the Colab/kagglehub cache), but running locally
  may require a Kaggle API token — see the
  [kagglehub docs](https://github.com/Kaggle/kagglehub) if you hit an
  authentication prompt.

## Installation

```bash
git clone <this-repo-url>
cd Pneumonia_Detection
pip install -r requirements.txt
```

## Usage

1. Open `ChestX-RayProject_123.ipynb` in Google Colab or Jupyter.
2. Run the cells top to bottom:
   - Cell 1 downloads the dataset via `kagglehub`.
   - Cells 2–4 build the `train`/`test` image datasets (224×224, batch size 32) and normalize pixel values.
   - Cell 5–6 build and compile the MobileNetV2-based model.
   - Cell 7 trains the model (5 epochs).
   - Cells 9–10 plot accuracy/loss curves.
   - Cell 11 lets you upload a custom X-ray image for a live prediction (Colab file upload widget).
   - Cell 12 prints a confusion matrix and classification report on the test set.

## Results

Trained for 5 epochs on 5,216 training images / 624 test images:

| Metric              | Value   |
|---------------------|---------|
| Final test accuracy | 81.57%  |
| Final training accuracy | 97.76% |

These numbers come directly from the notebook's saved run and will vary
slightly on re-training. The gap between training and validation accuracy
suggests the model would benefit from further regularization or fine-tuning
of the backbone — see below.

## Future Improvements

- Unfreeze and fine-tune the top layers of MobileNetV2 for a few epochs to close the train/val accuracy gap.
- Add data augmentation (rotation, zoom, flips) to reduce overfitting.
- Track experiments with more epochs and early stopping.
- Save the trained model (`.keras`/`.h5`) and add a standalone inference script outside of Colab.
- Add a proper train/validation/test split (the dataset's original validation split is very small).

## Screenshots

_Add training curves / sample predictions here._

```
docs/
└── accuracy_curve.png
└── sample_prediction.png
```

## License

Distributed under the [MIT License](./LICENSE).

## Author

**Abdullah**
