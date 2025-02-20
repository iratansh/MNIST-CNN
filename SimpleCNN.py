from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self):
        """
        Constructor for SimpleCNN.
        
        Creates a PyTorch nn.Module with the following architecture:
        
        Conv1: Conv2d(1, 32, kernel_size=3)
        ReLU
        MaxPool2d(kernel_size=2, stride=2)
        
        Conv2: Conv2d(32, 64, kernel_size=3)
        ReLU
        MaxPool2d(kernel_size=2, stride=2)
        
        Conv3: Conv2d(64, 128, kernel_size=3)
        ReLU
        MaxPool2d(kernel_size=2, stride=2)
        
        FC1: Linear(128 * 7 * 7, 128)
        Dropout(p=0.5)
        FC2: Linear(128, 10)
        """
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Activation and pooling
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 7 * 7, 128)  # Image inputs are (28 x 28)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)  

    def forward(self, x):
        # Convolutional block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Convolutional block 2
        x = self.conv2(x)
        x = self.relu(x)
        
        # Convolutional block 3
        x = self.conv3(x)
        x = self.pool(x)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Fully connected block
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

