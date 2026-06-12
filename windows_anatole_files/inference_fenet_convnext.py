import torch
import torch.nn.functional as F_nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from model_fenet_convnext import create_model



""""Ce script implémente une fonction d'inférence pour notre modèle de classification de fenêtres basé sur ConvNeXT.
Il charge le modèle entraîné, effectue l'inférence sur une image de test, et affiche les résultats en annotant l'image avec les boîtes de détection et les pourcentages d'ouverture calculés à partir des hauteurs des boîtes détectées.
Il inclut également une fonction pour appliquer les mêmes traitements d'expert (Padding carré + CLAHE) que ceux utilisés lors de l'entraînement, afin d'assurer une cohérence maximale entre les données d'entraînement et d'inférence, ce qui est crucial pour la performance du modèle.   """

def appliquer_traitement_expert(crop_img):
    """
    Applique le Padding et le CLAHE exactement comme lors de l'entraînement.
    C'est CRUCIAL pour que le modèle reconnaisse les mêmes motifs.
    """
    # 1. Padding Carré
    h, w = crop_img.shape[:2]
    max_side = max(h, w)
    toile = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    toile[(max_side-h)//2:(max_side-h)//2+h, (max_side-w)//2:(max_side-w)//2+w] = crop_img
    
    # 2. CLAHE (Local Contrast)
    lab = cv2.cvtColor(toile, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img_contrast = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    return cv2.resize(img_contrast, (224, 224))

def main(model_path, facade_path, liste_fenetres):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img_facade = cv2.imread(facade_path)
    img_visu = img_facade.copy()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Analyse de {len(liste_fenetres)} fenêtres sur la façade...")

    for (x, y, w, h) in liste_fenetres:
        # 1. Extraire le crop
        crop = img_facade[y:y+h, x:x+w]
        
        # 2. Appliquer les mêmes traitements que durant l'entraînement
        crop_expert = appliquer_traitement_expert(crop)
        
        # 3. Préparer pour ConvNeXt
        crop_rgb = cv2.cvtColor(crop_expert, cv2.COLOR_BGR2RGB)
        img_tensor = transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)

        # 4. Inférence
        with torch.no_grad():
            output = model(img_tensor)
            prob = F_nn.softmax(output, dim=1)[0]
            etat = "Ouvert" if prob[1] > prob[0] else "Ferme"
            conf = prob[1] if etat == "Ouvert" else prob[0]

        # 5. Dessiner sur la façade
        couleur = (0, 255, 0) if etat == "Ouvert" else (0, 0, 255)
        cv2.rectangle(img_visu, (x, y), (x + w, y + h), couleur, 2)
        cv2.putText(img_visu, f"{etat} {conf:.2f}", (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 2)

    cv2.imwrite("resultat_final_facade.jpg", img_visu)
    print("Analyse terminée ! Résultat : resultat_final_facade.jpg")

if __name__ == "__main__":
    # Liste de vos coordonnées de fenêtres détectées (x, y, w, h)
    # Dans le futur, ces coordonnées viendront de votre modèle Faster R-CNN
    mes_fenetres = [[50, 50, 150, 150], [250, 50, 150, 150]] 
    
    path_model = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\windows_anatole_files\convnextv2_expert_fenetres.pth"
    Image_path=r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\pso_files\resultat_resnet101.jpg"

    main(path_model, Image_path, mes_fenetres)