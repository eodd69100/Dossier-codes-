import torch
import cv2
from PIL import Image
from torchvision.transforms import functional as F

from model_pso_resnet101 import create_model

def detect_pso(model, image_path, threshold, device):
    """Détecte les PSO dans une image donnée en utilisant le modèle entraîné."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes  = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()

    keep = scores > threshold
    return boxes[keep], scores[keep]

def sorted_etage(coords, seuil_salle):
    """Sépare les boîtes détectées en deux étages (haut et bas) selon leur ordonnée ymin."""
    salle_haut, salle_bas = [], []
    for box in coords:
        _, ymin, _, _ = box
        if ymin < seuil_salle:  
            salle_haut.append(box)
        else:
            salle_bas.append(box)
    return salle_haut, salle_bas

def etat_ouverture(box, fenetre_height):
    """Calcule le pourcentage d'ouverture avec calibration pour les boîtes incomplètes."""
    _, ymin, _, ymax = box
    hauteur_boite = ymax - ymin
    ratio_couverture = hauteur_boite / fenetre_height
    
    # Calibration : Si le modèle couvre plus de 90% de la fenêtre, on force à 0% (Fermé)
    if ratio_couverture>=0.99:
        ouverture = 0.0
    else:
        ouverture = (1 - ratio_couverture) * 100
        
    return max(0.0, min(100.0, round(ouverture, 1)))


def main(model_path, image_test_path,seuil_salle,hauteur_fenetre_haut,hauteur_fenetre_bas):
    print("="*40)
    print("Configuration et Chargement du Modèle")
    print("="*40)

    device = torch.device("cpu") 
    model = create_model(num_classes=2, pretrained=False)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Modèle chargé avec succès !")
    except FileNotFoundError:
        print("Erreur : Le fichier est introuvable.")
        return

    model = model.to(device)
    model.eval()

    print("\nAnalyse de l'image en cours")
    boites, _ = detect_pso(model, image_test_path, threshold=0.6, device=device)
    print(f"Nombre d'objets détectés : {len(boites)}")

    image_originale = cv2.imread(image_test_path)
    if image_originale is None:
        print("Erreur : Impossible de lire l'image.")
        return

    image_annotee = image_originale.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    zones_texte_occupees = []
    
    # DESSIN SUR L'IMAGE
    coordonnees_boites = []
    for i in range(len(boites)):
        xmin, ymin, xmax, ymax = map(int, boites[i])
        coordonnees_boites.append(boites[i]) # On stocke la boîte brute pour le tri ultérieur
        
        # Choix de la bonne hauteur de référence pour le calcul
        if ymin < seuil_salle:
            h_ref = hauteur_fenetre_haut
        else:
            h_ref = hauteur_fenetre_bas
            
        opened = etat_ouverture(boites[i], h_ref)
        
        # Dessin de la boîte
        cv2.rectangle(image_annotee, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Préparation du texte
        texte_label = "opened: "
        texte_opened = f"{opened:.0f}%"
        
        taille_label, _ = cv2.getTextSize(texte_label, font, 0.5, 1)
        taille_score, _ = cv2.getTextSize(texte_opened, font, 0.5, 1)
        hauteur_texte = max(taille_label[1], taille_score[1])
        largeur_totale = taille_label[0] + taille_score[0]
        
        text_x, text_y = xmin, ymin - 5
        
        # Logique Anti-Superposition
        en_collision = True
        while en_collision:
            en_collision = False
            rect_actuel = (text_x, text_y - hauteur_texte, text_x + largeur_totale, text_y)
            for zone in zones_texte_occupees:
                zx1, zy1, zx2, zy2 = zone
                x1, y1, x2, y2 = rect_actuel
                if not (x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2):
                    en_collision = True
                    text_y -= (hauteur_texte + 5)
                    break
            if text_y - hauteur_texte < 0:
                text_y = ymax + hauteur_texte + 10
                en_collision = False 
                
        zones_texte_occupees.append((text_x, text_y - hauteur_texte, text_x + largeur_totale, text_y))
        
        # Affichage du texte
        cv2.putText(image_annotee, texte_label, (text_x, text_y), font, 0.5, (255, 0, 0), 1)
        cv2.putText(image_annotee, texte_opened, (text_x + taille_label[0], text_y), font, 0.5, (0, 0, 255), 1)

    # Sauvegarde de l'image
    
    
    nom_fichier_final = "resultat_resnet101.jpg"
    cv2.imwrite(nom_fichier_final, image_annotee)
    print(f"Image finale sauvegardée sous '{nom_fichier_final}'")

    print("\nAnalyse de l'état d'ouverture des PSO détectées (de gauche à droite) :")
    salle_haut, salle_bas = sorted_etage(coordonnees_boites, seuil_salle)
    salle_haut = sorted(salle_haut, key=lambda b: b[0])
    salle_bas = sorted(salle_bas, key=lambda b: b[0])
    for i, box in enumerate(salle_haut):
        print(f"PSO {i+1} (Haut) : {etat_ouverture(box, hauteur_fenetre_haut):.0f}%")
    for i, box in enumerate(salle_bas):
        print(f"PSO {i+1} (Bas)  : {etat_ouverture(box, hauteur_fenetre_bas):.0f}%")


if __name__ == "__main__":
    model_path = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\pso_files\faster_pso_resnet101_best.pth"
    image_test_path = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Dataset_Redressee_anatole\redressee_SYFW0256.JPG"
    imag_size = cv2.imread(image_test_path).shape
    seuil_salle = imag_size[0] // 2  # On considère que la salle est divisée en deux par la moitié de la hauteur de l'image
    hauteur_fenetre_haut =371
    hauteur_fenetre_bas =352
    
    main(model_path, image_test_path, seuil_salle, hauteur_fenetre_haut, hauteur_fenetre_bas)
    