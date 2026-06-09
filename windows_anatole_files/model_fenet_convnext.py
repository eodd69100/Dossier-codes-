import torch
import torch.nn as nn
import timm

def create_model(num_classes=2):
    model = timm.create_model('convnextv2_nano', pretrained=True)
    # On gèle tout le backbone (on ne modifie pas les poids déjà appris sur ImageNet)
    for param in model.parameters():
        param.requires_grad = False
        
    # On ne réentraîne que la dernière couche (la tête)
    num_features = model.head.fc.in_features
    model.head.fc = nn.Linear(num_features, num_classes) # On adapte la tête pour notre classification binaire (Ouvert vs Fermé)
    return model