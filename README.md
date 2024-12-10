# MNIST Classification with PyTorch

This project implements a CNN model for MNIST digit classification with the following specifications:
- Parameters: < 20k
- Test Accuracy: 99.42%
- Uses Batch Normalization, Dropout, and Global Average Pooling
- Trained for 15 epochs

## Model Architecture
- Input Layer: Conv2d(1, 8, 3)
- Multiple Conv blocks with BatchNorm and Dropout
- Global Average Pooling
- Final 1x1 Convolution layer

## Results
- Parameters: 19,866
- Best Test Accuracy: 99.42%
- Training completed in 15 epochs

## Project Structure
- `model.py`: CNN architecture
- `main.py`: Training and evaluation code
- `test_model.py`: Test cases for model validation
- `.github/workflows/model_tests.yml`: GitHub Actions workflow

## Training Logs 