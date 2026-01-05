# GANs Human Faces Generation

## Description
This project implements a Generative Adversarial Network (GAN) to generate grayscale human faces. It uses a custom dataset of human faces (downloaded from Kaggle), processes them (masking, cropping, converting to grayscale), and trains a Generator and Discriminator network. The model is trained to generate realistic-looking human faces from random noise.

## Directory Structure
- `gans-human-faces-generation.ipynb`: The main Jupyter Notebook that handles data downloading, processing, and model training.
- `Packages/`: Contains the model definitions and utility functions.
    - `generator.py`: Defines the Generator network architecture.
    - `discriminator.py`: Defines the Discriminator network architecture.
    - `Imgprocessing.py`: Image processing utilities, including face masking using MTCNN and custom transformations.
- `Drafts/`: Contains experimental scripts and notebooks.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- numpy
- opencv-python (cv2)
- facenet-pytorch (for MTCNN)
- PIL
- kagglehub (to download dataset)

## Usage
1.  **Install Dependencies:** Ensure you have the required packages installed.
2.  **Run the Notebook:** Open and run `gans-human-faces-generation.ipynb`.
    - The notebook will automatically download the dataset `mostafabakr8962/human-faces-dataset` using `kagglehub`.
    - It sets up the data loader with custom transformations (face masking, grayscale conversion, resizing).
    - Initializes the Generator and Discriminator models.
    - Trains the GAN.
    - Generates and saves images during training.

## Hyperparameters
- **Epochs:** 50
- **Batch Size:** 256
- **Learning Rate:** 0.0002
- **Latent Dimension:** 128
- **Image Size:** 256x256
- **Channels:** 1 (Grayscale)
- **Optimizer:** Adam (betas=(0.5, 0.999))

## Features
- **Face Masking:** Uses MTCNN to detect faces and applies a soft mask to focus on facial features and remove background noise.
- **Grayscale Generation:** Focuses on generating structural details without color complexity.
