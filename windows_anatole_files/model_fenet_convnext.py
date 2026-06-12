import torch
import torch.nn as nn
import timm

"""
Ce script définit la structure de notre modèle de classification de fenêtres basé sur ConvNeXT.
Il utilise la bibliothèque timm pour charger un modèle pré-entraîné sur ImageNet, et gèle les poids du backbone pour ne réentraîner que la tête de classification adaptée à notre tâche binaire (Ouvert vs Fermé).
Cette fonction create_model() est utilisée dans le script d'entraînement pour initialiser le modèle avant de le former sur notre dataset spécifique.
"""

def create_model(num_classes=2):
    model = timm.create_model('convnextv2_nano', pretrained=True)
    # On gèle tout le backbone (on ne modifie pas les poids déjà appris sur ImageNet)
    for param in model.parameters():
        param.requires_grad = False
        
    # On ne réentraîne que la dernière couche (la tête)
    num_features = model.head.fc.in_features
    model.head.fc = nn.Linear(num_features, num_classes) # On adapte la tête pour notre classification binaire (Ouvert vs Fermé)
    return model 