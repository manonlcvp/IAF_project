import torch.nn as nn
from torchvision import models

class MovieGenreClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MovieGenreClassifier, self).__init__()
        # Charger un modèle pré-entraîné, ici ResNet50 par exemple
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Gel des couches préalablement entraînées (si tu veux fine-tuner uniquement la dernière couche)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Remplacer la dernière couche pour la classification
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model(num_classes):
    """
    Charger et retourner un modèle avec le nombre de classes spécifié.
    """
    return MovieGenreClassifier(num_classes)

