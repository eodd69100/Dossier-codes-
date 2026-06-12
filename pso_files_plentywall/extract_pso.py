import os
import json
from PIL import Image

def main():

    """Ce script a pour objectif d'extraire automatiquement les PSOs (Petites Surfaces Ouvertes) à partir des images de façades présentes dans la base de données COCO.
    L'objectif est de découper précisément les PSOs en utilisant les coordonnées des bounding boxes fournies dans le fichier JSON d'annotations, 
    et de sauvegarder ces vignettes dans un dossier dédié pour création de datasets d'images de façades à partir de bâtiments vides)."""
    print("="*50)
    print(" EXTRACTION AUTOMATIQUE DES PSO ")
    print("="*50)

    # ==========================================
    # PARAMÈTRES (Chemins d'accès)
    # ==========================================
    # Chemin vers le JSON contenant les images à découper
    json_path = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\dataset_anatole_pso_2026.v1i.coco\train\_annotations.coco.json"
    
    # Dossier où se trouvent tes images originales de façades
    img_folder = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\dataset_anatole_pso_2026.v1i.coco\train\images"
    
    # Dossier où seront sauvegardés les PSOs découpés
    output_folder = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\pso_detoures"

    # Création du dossier de sortie s'il n'existe pas déjà
    os.makedirs(output_folder, exist_ok=True)

    # ==========================================
    # 2. CHARGEMENT DES DONNÉES COCO
    # ==========================================
    print(f"Chargement du fichier JSON : {json_path}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"\n[ERREUR] Le fichier {json_path} est introuvable. Vérifie le chemin.")
        return

    # Création d'un dictionnaire rapide pour retrouver le nom de l'image via son ID
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    print(f"-> {len(id_to_filename)} images trouvées dans le dictionnaire.")
    print(f"-> {len(data['annotations'])} PSOs à extraire.")

    # ==========================================
    # BOUCLE D'EXTRACTION
    # ==========================================
    print("\nDébut de l'extraction...")
    compteur_succes = 0
    compteur_erreurs = 0

    for ann in data['annotations']:
        img_id = ann['image_id']
        
        # Sécurité : vérifier que l'image existe dans la base "donneurs"
        if img_id not in id_to_filename:
            continue
            
        file_name = id_to_filename[img_id]
        img_path = os.path.join(img_folder, file_name)

        try:
            # 3.1 Ouvrir l'image originale
            image = Image.open(img_path).convert("RGB")

            # 3.2 Récupérer les coordonnées de la Bounding Box COCO
            # Format COCO : [x_min, y_min, largeur, hauteur]
            x, y, w, h = ann['bbox']

            # 3.3 Convertir pour PIL.crop() 
            # Format attendu par PIL : (gauche, haut, droite, bas)
            # On force en 'int' car les pixels ne peuvent pas avoir de virgule
            box = (int(x), int(y), int(x + w), int(y + h))

            # 3.4 Découper l'image
            pso_crop = image.crop(box)

            # 3.5 Sauvegarder la vignette
            # On utilise l'ID de l'image et l'ID de l'annotation pour éviter les doublons
            nom_sortie = f"pso_img{img_id}_ann{ann['id']}.jpg"
            chemin_sortie = os.path.join(output_folder, nom_sortie)
            
            pso_crop.save(chemin_sortie)
            compteur_succes += 1

        except FileNotFoundError:
            print(f"[Attention] Image introuvable sur le disque : {img_path}")
            compteur_erreurs += 1
        except Exception as e:
            print(f"[Erreur] Problème lors de la découpe de {file_name} : {e}")
            compteur_erreurs += 1

    # ==========================================
    # 4. BILAN DE L'OPÉRATION
    # ==========================================
    print("="*50)
    print(" BILAN DE L'EXTRACTION ")
    print("="*50)
    print(f" PSOs extraits avec succès : {compteur_succes}")
    if compteur_erreurs > 0:
        print(f" PSOs échoués : {compteur_erreurs}")
    print(f" Dossier de destination : {output_folder}")
    print("==================================================")

if __name__ == "__main__":
    main()