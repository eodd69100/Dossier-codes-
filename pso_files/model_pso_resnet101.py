
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator

"""
Ce script définit la fonction create_model() qui construit notre modèle de détection de PSO basé sur Faster R-CNN avec un backbone ResNet-101.
Il utilise des poids pré-entraînés sur ImageNet pour le backbone, et un générateur d'ancres personnalisé adapté à la détection de petites fenêtres et de PSO.
Cette fonction est utilisée dans le script d'entraînement pour initialiser le modèle avant de le former sur notre dataset spécifique."""




def create_model(num_classes, pretrained=True):
    # Choix des poids
    weights = ResNet101_Weights.DEFAULT if pretrained else None 
    
    # Création du backbone
    backbone = resnet_fpn_backbone(backbone_name='resnet101', weights=weights)
    
    # Générateur d'ancres personnalisé
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,)) # Ancre de base pour les petites fenêtres et les pso, puis des tailles plus grandes pour les fenêtres plus grandes
    aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 3.0, 4.0),) * len(anchor_sizes)
    custom_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    
    # Assemblage final
    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=custom_anchor_generator)
    
    return model   