import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from CustomMNISTDataset import CustomMNISTDataset
from SimpleCNN import SimpleCNN

def test_saved_model(model_path='mnist_cnn.pth',
                     test_csv_path='/Users/ishaanratanshi/MNIST Digit Recognizer Model/digit-recognizer/test.csv',
                     output_csv_path='submission.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Notes is_test=True only returns images
    test_dataset = CustomMNISTDataset(csv_path=test_csv_path, transform=transform, is_test=True)
    batch_size = 64
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Instantiate the model and load saved parameters
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode

    predictions = []
    # Disable gradient computation for inference
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images.float())
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy().tolist())
    
    # Create a DataFrame for submission
    submission_df = pd.DataFrame({
        'ImageId': list(range(1, len(predictions) + 1)),
        'Label': predictions
    })
    submission_df.to_csv(output_csv_path, index=False)
    print(f'Submission file saved to {output_csv_path}')

if __name__ == "__main__":
    test_saved_model()
