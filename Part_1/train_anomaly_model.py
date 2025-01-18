import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_anomaly import Autoencoder
import os

def train_anomaly_detector(data_dir, model_save_path, num_epochs=10, batch_size=32, lr=0.001):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Utilisation du GPU si possible

    # Transformation des images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standardisation
    ])

    # Chargement du dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Création du modèle autoencodeur
    model = Autoencoder().to(device)

    # Définition de la fonction de perte et de l'optimiseur
    criterion = torch.nn.MSELoss()  # Mesure la reconstruction
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Entraîner le modèle
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, _ in dataloader:  # Les labels sont ignorés

            inputs = inputs.to(device)
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, inputs)  # Comparer la sortie avec l'entrée

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Sauvegarde des poids du modèle
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":

    data_dir = '../data/part 1'  
    model_save_path = 'anomaly_detector.pth' 

    train_anomaly_detector(data_dir, model_save_path)
