import torch
import torch.nn as nn
from torchvision import models

class MovieRecommendationModel(nn.Module):
    def __init__(self, embedding_size=256):
        super(MovieRecommendationModel, self).__init__()
        
        # Charger un modèle pré-entraîné, ici ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Gel des paramètres du modèle ResNet (sauf la dernière couche)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Changer la couche de classification (fc) pour produire des embeddings
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_size)
        
        # Assurer que la couche de sortie (fc) peut être mise à jour pendant l'entraînement
        for param in self.resnet.fc.parameters():
            param.requires_grad = True  # Permet de mettre à jour la couche fc

    def forward(self, x):
        return self.resnet(x)

    def get_embedding(self, x):
        # Cette méthode renvoie les embeddings
        with torch.no_grad():  # Pas de calcul des gradients lors de l'extraction des embeddings
            return self.resnet(x)
