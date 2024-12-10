# MNIST Classification with PyTorch

This project implements a CNN model for MNIST digit classification with the following specifications:
- Parameters: 11,034 (well under 20k requirement)
- Test Accuracy: 99.42%
- Uses Batch Normalization, Dropout, and Global Average Pooling
- Trained for 15 epochs

## Model Architecture
- Input Layer: Conv2d(1, 8, 3)
- Multiple Conv blocks with BatchNorm and Dropout
- Global Average Pooling
- Final 1x1 Convolution layer

## Results
- Parameters: 11,034
- Best Test Accuracy: 99.42%
- Training completed in 15 epochs

## Project Structure
- `model.py`: CNN architecture definition
- `main.py`: Training and evaluation code
- `test_model.py`: Test cases for model validation
- `.github/workflows/model_tests.yml`: GitHub Actions workflow

## Training Logs

Epoch: 1
Epoch=1 Loss=0.1063 Batch=468: 100%|██████████████████████████████████████████████████████████████| 469/469 [01:08<00:00,  6.81it/s] 

Test set: Average loss: 0.1090, Accuracy: 9703/10000 (97.03%)


Epoch: 2
Epoch=2 Loss=0.0354 Batch=468: 100%|██████████████████████████████████████████████████████████████| 469/469 [01:06<00:00,  7.01it/s] 

Test set: Average loss: 0.0601, Accuracy: 9797/10000 (97.97%)


Epoch: 3
Epoch=3 Loss=0.0657 Batch=468: 100%|██████████████████████████████████████████████████████████████| 469/469 [01:15<00:00,  6.21it/s] 

Test set: Average loss: 0.0612, Accuracy: 9817/10000 (98.17%)


Epoch: 4
Epoch=4 Loss=0.0452 Batch=468: 100%|██████████████████████████████████████████████████████████████| 469/469 [01:18<00:00,  5.95it/s] 

Test set: Average loss: 0.0374, Accuracy: 9887/10000 (98.87%)


Epoch: 5
Epoch=5 Loss=0.0603 Batch=468: 100%|██████████████████████████████████████████████████████████████| 469/469 [01:09<00:00,  6.75it/s] 

Test set: Average loss: 0.0496, Accuracy: 9830/10000 (98.30%)


Epoch: 6
Epoch=6 Loss=0.0079 Batch=468: 100%|██████████████████████████████████████████████████████████████| 469/469 [01:06<00:00,  7.07it/s] 

Test set: Average loss: 0.0314, Accuracy: 9897/10000 (98.97%)


Epoch: 7
Epoch=7 Loss=0.0329 Batch=468: 100%|██████████████████████████████████████████████████████████████| 469/469 [01:24<00:00,  5.57it/s] 

Test set: Average loss: 0.0278, Accuracy: 9909/10000 (99.09%)


Epoch: 8
Epoch=8 Loss=0.0021 Batch=468: 100%|██████████████████████████████████████████████████████████████| 469/469 [01:09<00:00,  6.74it/s] 

Test set: Average loss: 0.0317, Accuracy: 9893/10000 (98.93%)


Epoch: 9
Epoch=9 Loss=0.0175 Batch=468: 100%|██████████████████████████████████████████████████████████████| 469/469 [01:10<00:00,  6.67it/s] 

Test set: Average loss: 0.0349, Accuracy: 9894/10000 (98.94%)


Epoch: 10
Epoch=10 Loss=0.0209 Batch=468: 100%|█████████████████████████████████████████████████████████████| 469/469 [01:11<00:00,  6.58it/s] 

Test set: Average loss: 0.0224, Accuracy: 9924/10000 (99.24%)


Epoch: 11
Epoch=11 Loss=0.0402 Batch=468: 100%|█████████████████████████████████████████████████████████████| 469/469 [01:07<00:00,  6.92it/s] 

Test set: Average loss: 0.0219, Accuracy: 9931/10000 (99.31%)


Epoch: 12
Epoch=12 Loss=0.0505 Batch=468: 100%|█████████████████████████████████████████████████████████████| 469/469 [01:16<00:00,  6.17it/s] 

Test set: Average loss: 0.0214, Accuracy: 9936/10000 (99.36%)

Epoch: 13
Epoch=13 Loss=0.0218 Batch=468: 100%|█████████████████████████████████████████████████████████████| 469/469 [01:14<00:00,  6.32it/s] 

Test set: Average loss: 0.0172, Accuracy: 9942/10000 (99.42%)


Epoch: 14
Epoch=14 Loss=0.0160 Batch=468: 100%|█████████████████████████████████████████████████████████████| 469/469 [01:14<00:00,  6.30it/s] 

Test set: Average loss: 0.0192, Accuracy: 9936/10000 (99.36%)

Epoch: 15
Epoch=15 Loss=0.0185 Batch=468: 100%|█████████████████████████████████████████████████████████████| 469/469 [01:17<00:00,  6.08it/s] 

Test set: Average loss: 0.0182, Accuracy: 9942/10000 (99.42%)