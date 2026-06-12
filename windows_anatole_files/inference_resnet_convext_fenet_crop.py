import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from torchvision import transforms
import torch.nn.functional as F_nn
import sys

# Chemins d'accès
sys.path.append(r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes")

# Imports de nos modèles
from pso_files.model_pso_resnet101 import create_model as create_detecteur_model
from model_fenet_convnext import create_model as create_expert_model

"""
Ce script implémente une fonction d'inférence pour notre pipeline de détection et classification de fenêtres.
Il charge à la fois le modèle de détection de PSO (Faster R-CNN avec ResNet-101) et le modèle de classification de fenêtres (ConvNeXT), effectue la détection des fenêtres sur une image de test, applique les mêmes traitements d'expert que ceux utilisés lors de l'entraînement, et affiche les résultats en annotant l'image avec les boîtes de détection et les pourcentages d'ouverture calculés à partir des hauteurs des boîtes détectées.
Il inclut également une fonction pour séparer les boîtes détectées en deux étages (haut et bas) selon leur position verticale, ainsi qu'une fonction de calibration pour ajuster les pourcentages d'ouverture en fonction de la hauteur des boîtes détectées.       
"""



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé pour le calcul : {device}")
def preprocess_crop(crop_img):
    """Padding carré + CLAHE pour l'expert ConvNeXt (224x224)"""
    h, w = crop_img.shape[:2]
    max_side = max(h, w)
    toile = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    toile[(max_side-h)//2:(max_side-h)//2+h, (max_side-w)//2:(max_side-w)//2+w] = crop_img
    
    # CLAHE
    lab = cv2.cvtColor(toile, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img_contrast = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.resize(img_contrast, (224, 224))

def detect_windows(model, image_path, threshold, device):
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).to(device)
    with torch.no_grad():
        outputs = model([img_tensor])[0]
    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    keep = scores > threshold
    return boxes[keep] 

def main(detecteur_path, expert_path, image_test_path):
    device = torch.device("cpu")
    
    # 1. Chargement des modèles
    detecteur = create_detecteur_model(num_classes=2) #
    detecteur.load_state_dict(torch.load(detecteur_path, weights_only=True)) # weights_only=True pour ne charger que les poids, pas la structure complète (utile si on a modifié la classe du modèle)
    detecteur.eval().to(device)# 
    
    expert = create_expert_model(num_classes=2)
    expert.load_state_dict(torch.load(expert_path, weights_only=True))
    expert.eval().to(device)
    
    # 2. Détection
    boites = detect_windows(detecteur, image_test_path, 0.5, device)
    image_originale = cv2.imread(image_test_path)
    image_annotee = image_originale.copy()
    
    trans_expert = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Traitement Expert par fenêtre
    for (xmin, ymin, xmax, ymax) in boites:
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        crop = image_originale[ymin:ymax, xmin:xmax]
        if crop.size == 0: continue
            
        crop_ready = preprocess_crop(crop)
        crop_tensor = trans_expert(Image.fromarray(cv2.cvtColor(crop_ready, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = expert(crop_tensor)
            prob = F_nn.softmax(output, dim=1)[0]
            etat = "Ouvert" if prob[1] > prob[0] else "Ferme"
            #if 0.5 < prob[1] < 0.6: etat = "Incertitude"
            conf = prob[1].item() if etat == "Ouvert" else prob[0].item() # 
            
        # Annotation
        couleur = (0, 255, 200) if etat == "Ouvert" else (0, 0, 255)
        cv2.rectangle(image_annotee, (xmin, ymin), (xmax, ymax), couleur, 2)
        cv2.putText(image_annotee, f"{etat} {conf*100:.0f}%", (xmin, ymin-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)

    cv2.imwrite("resultat_final_pipeline.jpg", image_annotee)
    print("Analyse terminée ! Image sauvegardée : 'resultat_final_pipeline.jpg'")

if __name__ == "__main__":
    detecteur_path = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\pso_files\faster_pso_resnet101_best.pth"
    expert_path = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\windows_anatole_files\convnextv2_expert_fenetres.pth"
    image_test_path = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\pso_files\resultat_resnet101.jpg"
    
    main(detecteur_path, expert_path, image_test_path)