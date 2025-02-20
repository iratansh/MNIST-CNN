import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kagglehub

class CustomMNISTDataset(Dataset):
    def __init__(self, csv_path, transform=None, is_test=False):
        """
        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing the data
        transform : callable, optional
            A function/transform that takes in a numpy array and returns a transformed version.
            E.g, ``transforms.ToTensor`` for turning the image to a Tensor.
        is_test : bool, optional
            Whether or not this is a test set. If true, doesn't include labels in the data frame.
        """
        
        self.data_frame = pd.read_csv(csv_path)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.data_frame)

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int
            Index of the item to be retrieved
            
        Returns
        -------
        (image, label) if not test set, else just image.
        image : PIL Image
            The image to be used for training/testing
        label : int
            The label associated with the image
        """
        item = self.data_frame.iloc[index]
        if self.is_test:
            image = item.values.reshape(28, 28).astype(np.uint8)
            label = None
        else:
            image = item[1:].values.reshape(28, 28).astype(np.uint8)
            label = item.iloc[0]
        
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image
        else:
            return image, label


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = CustomMNISTDataset(csv_path='/Users/ishaanratanshi/MNIST Digit Recognizer Model/digit-recognizer/train.csv', transform=transform, is_test=False)
    test_dataset = CustomMNISTDataset(csv_path='/Users/ishaanratanshi/MNIST Digit Recognizer Model/digit-recognizer/test.csv', transform=transform, is_test=True)

    print('Train Size: ', str(len(train_dataset)), ' Test Size: ', str(len(test_dataset)))
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    for example_data, example_labels in train_loader:
        example_image = example_data[0]
        print("Input Size:" , example_data.size())
        example_image_numpy = example_image.permute(1, 2, 0).numpy()
        plt.imshow(example_image_numpy)
        plt.title(f"Label: {example_labels[0]}")
        plt.show()
        break
