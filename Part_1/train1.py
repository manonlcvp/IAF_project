import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model1 import load_model
import os

def train_model(data_dir, model_save_path, num_classes, num_epochs=10, batch_size=32, lr=0.001):
    """
    Objectif = Entraîner le modèle sur un dataset d'images et sauvegarder les poids du modèle.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # utilisation du GPU si possible

    # Transformation des images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # valeurs de moyenne et d'ecart type standardisees pour un modele pre-entraine sur ImageNet
    ])
    
    # Chargement du dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Création du modèle
    model = load_model(num_classes, device)
    
    # Définition de la fonction de perte et l'optimiseur
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Entraîner le modèle
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in dataloader:

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Prediction
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Affichage de la fonction de perte et de la precision
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        epoch_loss = running_loss / len(dataloader)
        accuracy = correct_preds / total_preds * 100
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Sauvegarde des poids du modèle
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":

    data_dir = 'data/part 1'  # Chemin vers le dataset
    model_save_path = 'prediction_model.pth'
    num_classes = 10  # 10 categories de films
    
    train_model(data_dir, model_save_path, num_classes)