import torch
from tqdm import tqdm

"""
Ce script implémente les fonctions d'entraînement et d'évaluation pour notre modèle de classification de fenêtres basé sur ConvNeXT.
Il inclut une fonction train_one_epoch() qui exécute une époque complète d'apprentissage (passage avant + rétropropagation), et une fonction evaluate_loss()
 qui évalue le modèle sur un jeu de validation sans modifier les gradients.
Ces fonctions sont utilisées dans le script principal pour entraîner le modèle et suivre l'évolution de la perte"""

def train_one_epoch(model, optimizer, criterion, data_loader, device):
    """Exécute une époque complète d'apprentissage (Passage avant + Rétropropagation)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(data_loader, desc="Entraînement", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Phase de mise à jour des poids synaptiques
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calcul des métriques en direct
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total

def evaluate_loss(model, criterion, data_loader, device):
    """Évalue le modèle sur le jeu de validation sans modifier les gradients."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total