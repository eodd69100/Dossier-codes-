import torch
import cv2
from PIL import Image
from torchvision.transforms import functional as F

from Codes.windows_anatole_files.model_fenet_convnext import create_model

def detect_windows(model, image_path, threshold, device):
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes  = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()

    keep = scores > threshold
    return boxes[keep], scores[keep], labels[keep]

def sorted_etage(donnees_completes, seuil_salle):
    # donnees_completes contient des paquets de données : (box, score, label)
    # Cela permet de ne pas perdre le label quand on trie les étages !
    salle_haut, salle_bas = [], []
    for data in donnees_completes:
        box = data[0] # La boîte est le premier élément du paquet
        _, ymin, _, _ = box
        if ymin < seuil_salle:  
            salle_haut.append(data)
        else:
            salle_bas.append(data)
    return salle_haut, salle_bas

def main(model_path, image_test_path):
    print("="*40)
    print("Configuration et Chargement du Modèle")
    print("="*40)

    device = torch.device("cpu") 
    
    # 3 classes (0: Fond, 1: Fermé, 2: Ouvert)
    model = create_model(num_classes=3, pretrained=False)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Modèle chargé avec succès !")
    except FileNotFoundError:
        print("Erreur : Le fichier est introuvable.")
        return

    model = model.to(device)
    model.eval()

    print("\nAnalyse de l'image en cours...")
    boites, scores, labels = detect_windows(model, image_test_path, threshold=0.75, device=device)
    print(f"Nombre de fenêtres détectées : {len(boites)}")

    image_originale = cv2.imread(image_test_path)
    if image_originale is None:
        print("Erreur : Impossible de lire l'image.")
        return

    image_annotee = image_originale.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    zones_texte_occupees = []
    
    # Liste pour stocker (coordonnées, score, label) ensemble
    donnees_detectees = []

    for i in range(len(boites)):
        xmin, ymin, xmax, ymax = map(int, boites[i])
        score = scores[i] * 100
        label_id = labels[i]
        
        # On sauvegarde toutes les infos ensemble pour le tri par étage
        donnees_detectees.append(((xmin, ymin, xmax, ymax), score, label_id))
        
        # Choix du texte et de la couleur selon la classe
        if label_id == 1:
            texte_label = "Fermee: "
            couleur = (0, 0, 255) # Rouge
        else:
            texte_label = "Ouverte: "
            couleur = (0, 255, 0) # Vert
            
        cv2.rectangle(image_annotee, (xmin, ymin), (xmax, ymax), couleur, 2)
        
        texte_score = f"{score:.1f}%"
        taille_label, _ = cv2.getTextSize(texte_label, font, 0.5, 1)
        taille_score, _ = cv2.getTextSize(texte_score, font, 0.5, 1)
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
        
        # Dessin du texte de la même couleur que la boîte
        cv2.putText(image_annotee, texte_label, (text_x, text_y), font, 0.5, couleur, 1)
        cv2.putText(image_annotee, texte_score, (text_x + taille_label[0], text_y), font, 0.5, couleur, 1)

    nom_fichier_final = "resultat_fenetres_jh_resnet101.jpg"
    cv2.imwrite(nom_fichier_final, image_annotee)
    print(f"Image finale sauvegardée sous '{nom_fichier_final}'")

    print("\nAnalyse de l'état des fenêtres (de gauche à droite) :")
    salle_haut, salle_bas = sorted_etage(donnees_detectees, seuil_salle=80)
    
    # Tri par coordonnée X (gauche à droite)
    salle_haut = sorted(salle_haut, key=lambda d: d[0][0])
    salle_bas = sorted(salle_bas, key=lambda d: d[0][0])
    
    for i, data in enumerate(salle_haut):
        etat = "Fermée " if data[2] == 1 else "Ouverte"
        print(f"Fenêtre {i+1} (Haut) : {etat} (Confiance: {data[1]:.1f}%)")
        
    for i, data in enumerate(salle_bas):
        etat = "Fermée " if data[2] == 1 else "Ouverte"
        print(f"Fenêtre {i+1} (Bas)  : {etat} (Confiance: {data[1]:.1f}%)")

if __name__ == "__main__":
    model_path = r""
    image_test_path = r""
    
    main(model_path, image_test_path)