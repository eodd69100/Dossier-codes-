# main_fenet_convnext.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import cv2
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

# Importation de nos briques modulaires locales
from data_fenet_cropee import get_dataloaders
from model_fenet_convnext import create_model
from engine_fenet_convnext import train_one_epoch, evaluate_loss
from graph_fenet_connext import generer_courbes, generer_matrice_confusion, generer_courbe_pr


# CLASSE GRAD-CAM SUR MESURE POUR CONVNEXT

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks pour capturer les calculs en direct
        self.target_layer.register_forward_hook(self.save_activation) # Hook pour capturer les activations de la couche cible
        self.target_layer.register_full_backward_hook(self.save_gradient) # Hook pour capturer les gradients lors du passage arrière (backward)

    def save_activation(self, module, input, output):
        self.activations = output.detach() # On stocke les activations de la couche cible pour les utiliser dans la génération de la heatmap

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generer_heatmap(self, input_tensor, class_idx=None):
        # Passage avant (Forward)
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Passage arrière (Backward) pour obtenir les gradients
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        # Calcul de la carte d'attention
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3]) # Moyenne des gradients sur les dimensions spatiales (H, W)
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(self.activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0) # ReLU (on ne garde que l'attention positive)
        
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap) # Normalisation entre 0 et 1
            
        return heatmap, class_idx


# FONCTION PRINCIPALE

def main(DOSSIER_UNIQUE_CLASSIF):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Status : Entraînement de ConvNeXt V2 lancé sur l'appareil ──► [{device}]")

    train_loader, val_loader = get_dataloaders(DOSSIER_UNIQUE_CLASSIF, batch_size=2, val_split=0.2)
    model = create_model(num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.009)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.1)

    num_epoches = 20
    meilleure_perte = float('inf')
    
    patience_early_stopping = 5
    compteur_early_stopping = 0

    historique_train_loss = []
    historique_val_loss = []

    print("\n Lancement de la boucle d'apprentissage géométrique")
    print("-" * 80)

    for epoch in range(num_epoches):
        train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader, device)
        val_loss, val_acc = evaluate_loss(model, criterion, val_loader, device)

        historique_train_loss.append(train_loss)
        historique_val_loss.append(val_loss)

        print(f"Époque [{epoch+1:02d}/{num_epoches:02d}] | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.1f}%")

        scheduler.step(val_loss)

        if val_loss < meilleure_perte:
            meilleure_perte = val_loss
            compteur_early_stopping = 0
            torch.save(model.state_dict(), "convnextv2_expert_fenetres.pth")
            print("[SAUVEGARDE] Nouveau point de contrôle optimal enregistré.")
        else:
            compteur_early_stopping += 1
            print(f"Aucune amélioration de la perte de validation depuis {compteur_early_stopping} époque(s).")

        print("-" * 80)
        
        if compteur_early_stopping >= patience_early_stopping:
            print(f"\n EARLY STOPPING DÉCLENCHÉ : Arrêt de l'entraînement à l'époque {epoch+1}.")
            break

    print("\n Entraînement terminé. Génération des rapports d'évaluation visuels")

    
    # ÉVALUATION FINALE (GÉNÉRATION DES GRAPHES POUR LE RAPPORT)
    
    generer_courbes(historique_train_loss, historique_val_loss, save_path="rapport_loss_convnext.png")

    model.load_state_dict(torch.load("convnextv2_expert_fenetres.pth", map_location=device))
    model.eval()

    y_vraies = []
    y_predites = []
    y_scores = []

    print("Analyse du jeu de validation pour la Matrice de Confusion et la Courbe PR...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1) # On convertit les sorties brutes en probabilités pour la classe "Ouvert"
            y_scores.extend(probs[:, 1].cpu().numpy())
            _, preds = torch.max(outputs, 1)
            y_vraies.extend(labels.cpu().numpy())
            y_predites.extend(preds.cpu().numpy())

    generer_matrice_confusion(y_vraies, y_predites, classes=["Fermé", "Ouvert"], save_path="rapport_heatmap_convnext.png")
    generer_courbe_pr(y_vraies, y_scores, save_path="rapport_pr_curve_convnext.png")

    
    # GÉNÉRATION DE L'IMAGE GRAD-CAM
   
    print("Génération de l'explicabilité Grad-CAM")
    
    # On cible la dernière couche de blocs de ConvNeXt V2 (dans la bibliothèque timm)
    target_layer = model.stages[-1].blocks[-1]
    cam = GradCAM(model, target_layer)

    # On prend la première image du DataLoader de validation
    images, labels = next(iter(val_loader))
    img_tensor = images[0].unsqueeze(0).to(device) # On isole 1 image
    vrai_label = labels[0].item()

    # On génère la carte de chaleur
    # Il faut réactiver temporairement les gradients pour Grad-CAM
    for param in model.parameters():
        param.requires_grad = True
    
    heatmap, pred_idx = cam.generer_heatmap(img_tensor)

    # On "dénormalise" l'image pour la réafficher en couleurs réelles (OpenCV BGR)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    img_rgb = (images[0].to(device) * std + mean).permute(1, 2, 0).cpu().numpy()
    img_rgb = np.clip(img_rgb, 0, 1)
    img_bgr = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # On superpose la heatmap sur l'image
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)

    # Sauvegarde avec annotation
    etat_vrai = "Ouvert" if vrai_label == 1 else "Ferme"
    etat_pred = "Ouvert" if pred_idx == 1 else "Ferme"
    cv2.putText(superimposed_img, f"Vrai:{etat_vrai} | Pred:{etat_pred}", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imwrite("rapport_gradcam_convnext.png", superimposed_img)
    print("Tous les graphiques et le Grad-CAM ont été générés avec succès !")

if __name__ == "__main__":
    DOSSIER_UNIQUE_CLASSIF = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\dataset_classification"
    main(DOSSIER_UNIQUE_CLASSIF)