This Python program implements a convolutional neural network (CNN) to classify MNIST digit images. Score = 0.98900

### Overview
- **Custom Data Handling:**  
  The code uses a custom dataset class (`CustomMNISTDataset`) to load MNIST data from CSV files for both training and testing. It applies data augmentation and preprocessing using a composition of transformations (random rotation, tensor conversion, and normalization).

- **Model Architecture:**  
  The CNN model is defined externally in the `SimpleCNN` module. The model is instantiated and moved to a device (GPU if available, otherwise CPU).

- **Training Pipeline:**  
  The training loop iterates over a specified number of epochs (100) and processes the data in batches (batch size of 64). For each batch, it performs a forward pass, computes the cross-entropy loss, backpropagates the error, and updates the model parameters using stochastic gradient descent (SGD) with momentum. The training progress is periodically printed.

- **Model Persistence:**  
  After training, the model’s learned parameters are saved to a file (`mnist_cnn.pth`) for later use.

### Technologies Used
- **Python:**  
  The programming language used for implementation.

- **PyTorch:**  
  A deep learning framework for constructing and training neural networks. It provides modules for building models (`torch.nn`), optimizers (`torch.optim`), and device management.

- **Torchvision:**  
  Utilized here for data transformation utilities (e.g., random rotations, normalization) to preprocess the MNIST images.

- **Custom Modules:**  
  - `CustomMNISTDataset`: Handles the loading and preprocessing of the MNIST data from CSV files.
  - `SimpleCNN`: Defines the architecture of the CNN model used for digit classification.


Overall, the code demonstrates a standard deep learning workflow—from data preparation and model definition to training and saving the model—using popular Python libraries for machine learning and computer vision.

Citations:
https://www.kaggle.com/c/digit-recognizer/data
https://www.youtube.com/watch?v=2w0pRriQG3A&ab_channel=Ryan%26MattDataScience