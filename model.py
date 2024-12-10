import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  # 28x28x8
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        )
        
        # CONV Block 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),  # 28x28x16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )
        
        # Transition Block 1
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14x16
        
        # CONV Block 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),  # 14x14x16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )
        
        # CONV Block 3
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),  # 14x14x16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        )
        
        # Transition Block 2
        self.pool2 = nn.MaxPool2d(2, 2)  # 7x7x16
        
        # CONV Block 4
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 32, 3),  # 5x5x32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1)
        )
        
        # Global Average Pooling
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)  # 1x1x32
        )
        
        # Final Fully Connected Layer
        self.fc = nn.Conv2d(32, 10, 1)  # 1x1x10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1) 