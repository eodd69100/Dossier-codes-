import os
from PIL import Image
from torchvision.transforms import v2

def generer_images_pieges(chemin_image_source, dossier_sortie="images_test_ia"):
    """
    Prend une image de façade et génère 5 variations extrêmes pour tester l'IA.
    """
    # 1. Création du dossier de sortie s'il n'existe pas
    os.makedirs(dossier_sortie, exist_ok=True)
    
    # 2. Chargement de l'image de base
    img_base = Image.open(chemin_image_source).convert("RGB")
    nom_base = os.path.basename(chemin_image_source).split('.')[0]
    
    print(f"Génération de variantes pour : {nom_base}...")

    # 3. Nos 5 "pièges" pour l'IA
    transformations = {
        "1_jour_nuit": v2.ColorJitter(brightness=(0.2, 2.0), contrast=0.8), # Très sombre ou très clair
        "2_brouillard": v2.GaussianBlur(kernel_size=11, sigma=5.0), # Très flou (caméra sale ou brouillard)
        "3_perspective": v2.RandomPerspective(distortion_scale=0.4, p=1.0), # Photo prise de biais / d'en bas
        "4_rotation": v2.RandomRotation(degrees=25), # Photo prise de travers
        "5_bruit_couleur": v2.ColorJitter(saturation=2.0, hue=0.3) # Couleurs artificielles / mauvais capteur
    }

    # 4. Application et sauvegarde
    for nom_piege, transfo in transformations.items():
        img_variante = transfo(img_base)
        chemin_sauvegarde = os.path.join(dossier_sortie, f"{nom_base}_{nom_piege}.png")
        
        # Sauvegarde au format PNG comme demandé
        img_variante.save(chemin_sauvegarde, format="PNG")
        print(f" -> Créée : {chemin_sauvegarde}")

    print(f"\nTerminé ! 5 images PNG ont été créées dans le dossier '{dossier_sortie}'.")

# ==========================================
# LANCEMENT DU SCRIPT
# ==========================================
if __name__ == "__main__":
    # Remplacez par le bon chemin vers votre image si elle n'est pas dans le même dossier
    chemin_image = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Continuous_PSO.v5i.coco\test\images\redressee_SYFW2351_JPG.rf.40caef5d711206ea89e41b2a0d4e5d22.jpg" 
    
    if os.path.exists(chemin_image):
        generer_images_pieges(chemin_image)
    else:
        print(f"Erreur : L'image {chemin_image} est introuvable.")