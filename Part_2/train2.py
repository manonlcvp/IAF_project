import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from model2 import MovieRecommendationModel

class MoviePosterDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Dataset pour charger les affiches de films sans sous-dossiers.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]  # Retourner aussi le nom du fichier, utile pour la recommandation

def train_model(data_dir, model_save_path, num_epochs=10, batch_size=32, lr=0.001):
    """
    Objectif = Entraîner un modèle pour générer des embeddings d'affiches et sauvegarder les poids du modèle.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Utilisation du GPU si possible

    # Transformation des images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Valeurs de moyenne et d'écart-type standards pour ImageNet
    ])
    
    # Chargement du dataset personnalisé
    dataset = MoviePosterDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Création du modèle
    model = MovieRecommendationModel()
    model.to(device)  # Envoi du modèle sur le device (GPU ou CPU)
    
    # Définition de l'optimiseur
    optimizer = optim.Adam(model.resnet.fc.parameters(), lr=lr)  # Optimisation sur la couche fc uniquement

    # Entraînement du modèle
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, _ in dataloader:  # On ignore les labels car c'est de la recommandation
            inputs = inputs.to(device)
            optimizer.zero_grad()

            # Prédiction (embedding)
            embeddings = model(inputs)

            # Calcul de la perte
            loss = embeddings.norm(p=2, dim=1).mean()  # Minimiser la norme des embeddings
            loss.backward()  # Calcul des gradients
            optimizer.step()  # Mise à jour des poids

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Sauvegarde des poids du modèle
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    data_dir = 'data/part 2'  # Chemin vers le dataset
    model_save_path = 'recommendation_model.pth'
    
    train_model(data_dir, model_save_path)
