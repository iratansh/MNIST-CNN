import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from CustomMNISTDataset import CustomMNISTDataset
from SimpleCNN import SimpleCNN

class MNISTClassifier:
    def __init__(self):
        """
        Constructor for MNISTClassifier.

        Loads the MNIST dataset, transforms images to grayscale tensors, and creates a PyTorch DataLoader.
        Creates a SimpleCNN model, sets the loss criterion to CrossEntropyLoss, and sets the optimizer to Stochastic Gradient Descent.
        Sets the number of epochs to 100.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.train_dataset = CustomMNISTDataset(csv_path='/Users/ishaanratanshi/MNIST Digit Recognizer Model/digit-recognizer/train.csv', transform=transform, is_test=False)
        self.test_dataset = CustomMNISTDataset(csv_path='/Users/ishaanratanshi/MNIST Digit Recognizer Model/digit-recognizer/test.csv', transform=transform, is_test=True)
        self.batch_size = 64
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        self.model = SimpleCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.epochs = 100
        self.running_loss = 0.0

    def train(self):
        """
        Train the model on the MNIST dataset.

        Prints the loss after every 100 batches.
        """
        for epoch in range(self.epochs):
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs.float())
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.running_loss += loss.item()
                if i % 100 == 99:
                    print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {self.running_loss / 100:.3f}')
                    self.running_loss = 0.0
        print('Finished Training')

if __name__ == "__main__":
    mnist = MNISTClassifier()
    mnist.train()
    torch.save(mnist.model.state_dict(), 'mnist_cnn.pth')
