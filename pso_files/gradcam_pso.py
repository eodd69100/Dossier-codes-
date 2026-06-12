"""
gradcam_pso.py
--------------
Visualisation Grad-CAM spécifiquement corrigée pour Faster R-CNN.
Permet d'analyser les zones d'attention du modèle pour la détection de PSO.
"""

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms import functional as F_vision

# Importation de votre constructeur de modèle
from model_pso_resnet101 import create_model

"""
Ce script implémente une version corrigée de Grad-CAM adaptée à Faster R-CNN, qui prend en compte les spécificités de ce modèle 
(notamment la présence de plusieurs PSO détectés dans une même image). Il génère des cartes de chaleur pour les zones d'attention du modèle, en se basant sur la somme des scores de tous les PSO détectés au-dessus d'un seuil de confiance.
Il inclut également des fonctions utilitaires pour la superposition de la heatmap sur l'image originale et le dessin des boîtes de détection, avec une visualisation claire et informative."""



# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Adaptez uniquement ces chemins
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH   = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\pso_files\faster_pso_resnet101_best.pth"
IMAGES_DIR   = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\essai\Dataset_Redresse"
OUTPUT_DIR   = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\pso_files\gradcam_results"

NUM_CLASSES  = 2
SCORE_THRESH = 0.30   # Seuil de confiance minimal
TARGET_LAYER = "layer3" # CRUCIAL : layer3 est idéal pour les objets moyens/petits comme les fenêtres

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE GRAD-CAM (Corrigée pour Faster R-CNN)
# ══════════════════════════════════════════════════════════════════════════════
class FasterRCNNGradCAM:
    def __init__(self, model, target_layer_name="layer3"):
        self.model = model
        self.activations = None
        self.gradients   = None
        self._hook_layer(target_layer_name)

    def _hook_layer(self, layer_name):
        # On s'accroche au backbone ResNet
        layer = getattr(self.model.backbone.body, layer_name)

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def __call__(self, image_tensor):
        self.model.eval()
        self.model.zero_grad()

        # 1. Passage avec calcul des gradients activé
        with torch.enable_grad():
            preds_grad = self.model(image_tensor)

        # 2. Détachement pour l'affichage (évite les erreurs de graphe lors du dessin)
        preds = {k: v.detach().cpu() for k, v in preds_grad[0].items()}

        if len(preds["scores"]) == 0:
            return None, preds

        # 3. CORRECTION MAJEURE : On somme les scores de TOUS les PSO fiables
        scores = preds_grad[0]["scores"]
        target_score = torch.tensor(0.0, device=image_tensor.device, requires_grad=True)
        
        valid_detections = 0
        for score in scores:
            if score > SCORE_THRESH:
                target_score = target_score + score
                valid_detections += 1
                
        if valid_detections == 0:
            return None, preds # Aucun PSO n'a dépassé le seuil

        # 4. Rétropropagation sur la somme totale
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            print("  ⚠️ Erreur : Impossible de capturer les gradients.")
            return None, preds

        # 5. Calcul de la heatmap
        # Global Average Pooling sur les gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Pondération des activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam) # On ne garde que l'influence positive
        
        cam = cam.squeeze().detach().cpu().numpy()
        
        # Normalisation robuste [0, 1]
        if cam.max() > 0:
            cam -= cam.min()
            cam /= cam.max()
        else:
            cam = np.zeros_like(cam)

        # Nettoyage mémoire
        self.gradients = None
        self.activations = None

        return cam, preds


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_transform():
    # Faster R-CNN gère sa propre normalisation en interne.
    # On se contente de convertir l'image en tenseur [0, 1] comme en inférence classique.
    return lambda img: F_vision.to_tensor(img)

def overlay_heatmap(image_np, cam, alpha=0.5):
    H, W = image_np.shape[:2]
    cam_resized = cv2.resize(cam, (W, H))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * image_np).astype(np.uint8)
    return overlay

def draw_boxes(ax, preds, score_thresh):
    """Dessine les boîtes de détection sur un axe matplotlib."""
    for box, score in zip(preds["boxes"], preds["scores"]):
        if score.item() < score_thresh:
            continue
        
        x1, y1, x2, y2 = box.numpy()
        w_box, h_box = x2 - x1, y2 - y1
        color = "lime" if score.item() >= 0.75 else "orange"
        
        rect = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, max(y1 - 4, 0), f"PSO {score.item()*100:.0f}%", 
                color=color, fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

def visualize_and_save(image_np, overlay, preds, score_thresh, save_path, img_name):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Analyse d'Explicabilité (Grad-CAM) — {img_name}", fontsize=15, fontweight="bold")

    axes[0].imshow(image_np)
    axes[0].set_title("1. Image Originale + Détections", fontsize=12)
    axes[0].axis("off")
    draw_boxes(axes[0], preds, score_thresh)

    axes[1].imshow(overlay)
    axes[1].set_title(f"2. Carte de Chaleur (Couche: {TARGET_LAYER})", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("3. Superposition", fontsize=12)
    axes[2].axis("off")
    draw_boxes(axes[2], preds, score_thresh)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PROGRAMME PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Lancement du diagnostic Grad-CAM sur l'appareil ──► [{device}]")

    # 1. Chargement du modèle
    print(f"\nChargement du détecteur ResNet101...")
    model = create_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    
    gradcam = FasterRCNNGradCAM(model, target_layer_name=TARGET_LAYER)
    transform = get_transform()

    # 2. Préparation des images
    extensions = (".jpg", ".jpeg", ".png", ".bmp")
    images_list = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(extensions)]
    
    # On limite aux 10 premières images pour ne pas surcharger la machine
    images_list = images_list[:10] 
    print(f"Analyse de {len(images_list)} images ciblées...\n")

    # 3. Boucle d'analyse
    for img_name in images_list:
        img_path = os.path.join(IMAGES_DIR, img_name)
        
        pil_img    = Image.open(img_path).convert("RGB")
        img_np     = np.array(pil_img)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        img_tensor.requires_grad_() # Sécurité pour forcer le graphe

        print(f" ⏳ Traitement en cours : {img_name}...")
        cam, preds = gradcam(img_tensor)

        if cam is None:
            print("    ➔ Ignoré (Aucune détection fiable).")
            continue

        nb_detections = (preds["scores"] >= SCORE_THRESH).sum().item()
        print(f"    ➔ {nb_detections} PSO détecté(s). Génération de la Heatmap...")

        overlay   = overlay_heatmap(img_np, cam, alpha=0.5)
        save_name = os.path.splitext(img_name)[0] + "_gradcam_result.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        
        visualize_and_save(img_np, overlay, preds, SCORE_THRESH, save_path, img_name)

    print(f"\n✅ Diagnostic terminé. Les images sont disponibles dans : {OUTPUT_DIR}")

if __name__ == "__main__":
    main()