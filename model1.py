import torch.nn as nn
from torchvision import models

class MovieGenreClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MovieGenreClassifier, self).__init__()

        # Chargement du modèle pré-entraîné ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Gel des couches préalablement entraînées
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Remplacement de la dernière couche pour la classification
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model(num_classes, device):
    """
    Objectif = Charger et retourner un modèle avec le nombre de classes spécifié.
    """
    model = MovieGenreClassifier(num_classes)
    model.to(device)  # Envoie le modèle sur le device (GPU ou CPU)
    print(f"Le modèle est envoyé sur : {device}")
    return model

