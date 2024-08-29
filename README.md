# Jellyfish Classification

This repository contains code to train and evaluate a simple Convolutional Neural Network (CNN) for image classification using PyTorch. The network is designed to classify images from a custom dataset, and it includes training, testing, and visualization of the training progress.

## Requirements

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The dataset should be organized in the following structure:

```
- archive/
    - class1/
        - img1.jpg
        - img2.jpg
        ...
    - class2/
        - img1.jpg
        - img2.jpg
        ...
- test/
    - class1/
        - img1.jpg
        - img2.jpg
        ...
    - class2/
        - img1.jpg
        - img2.jpg
        ...
```

The dataset should be placed in the `archive/` directory for training and in the `test/` directory for testing.

## Model Architecture

The CNN model (`SimpleCNN`) consists of the following layers:
- 3 Convolutional layers with ReLU activations and max-pooling
- 2 Fully connected layers
- Dropout for regularization

The model is designed to classify images into `num_classes` categories.

## Training

The training loop involves:
- Data augmentation techniques to improve generalization
- Calculation of mean and standard deviation for dataset normalization
- Loss and accuracy computation for each epoch

Training runs for 30 epochs by default, with a learning rate of 0.001 using the Adam optimizer.

## Visualization

After training, the loss and accuracy for both the training and test datasets are plotted to visualize the model's performance over the epochs.

## Saving the Model

The trained model is saved as `YYYY-MM-DD_model.pth` where `YYYY-MM-DD` is the current date.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/esadaltunel/jellyfish_classification/tree/main
   ```
2. Navigate to the directory:
   ```bash
   cd SimpleCNN-Image-Classification
   ```
3. Run the training script:
   ```bash
   python train.py
   ```

The script will automatically train the model, evaluate it, and save the results.

## Results

- Loss and accuracy curves are displayed at the end of training.
- The final model is saved for future inference or further fine-tuning.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
