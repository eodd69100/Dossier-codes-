import os
import json
import random
from PIL import Image
import shutil


print("="*50)
print(" Génération et Fusion du Dataset Final")
print("="*50)

# ==========================================
# 1. PARAMÈTRES
# ==========================================
# Dossiers sources
dossier_psos = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\pso_detoures"
dossier_fonds = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\batiment_vide" # Les murs téléchargés sur internet
dossier_vraies_images = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\dataset_anatole_pso_2026.v1i.coco\train\images"
json_vraies_images = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\dataset_anatole_pso_2026.v1i.coco\train\_annotations.coco.json"

# Dossiers et fichiers de sortie (Le dataset final prêt pour l'entraînement)
dossier_sortie_images = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\images_artificielles"
json_sortie = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\train_artificiel.json"

NOMBRE_IMAGES_ARTIFICIELLES = 50 
MARGE_BORD = 40 # Les PSOs ne s'approcheront pas à moins de 40 pixels des bords

os.makedirs(dossier_sortie_images, exist_ok=True)

# ==========================================
# 2. PRÉPARATION DU DATASET FINAL
# ==========================================
print("1. Chargement des vraies façades préservées...")
with open(json_vraies_images, 'r') as f:
    data_final = json.load(f)

# On récupère la taille standard de tes images caméra (à partir de la première image)
standard_w = data_final["images"][0]["width"]
standard_h = data_final["images"][0]["height"]
print(f"-> Résolution standard détectée : {standard_w}x{standard_h} pixels.")

# Copie des vraies images dans le dossier final
for img_info in data_final["images"]:
    src = os.path.join(dossier_vraies_images, img_info["file_name"])
    dst = os.path.join(dossier_sortie_images, img_info["file_name"])
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)

print(f"-> {len(data_final['images'])} vraies images copiées dans le dossier final.")

# ==========================================
# 3. FONCTION ANTI-COLLISION
# ==========================================
def verifier_collision(nouvelle_boite, boites_existantes, marge=20):
    nx_min, ny_min, nx_max, ny_max = nouvelle_boite
    for (ex_min, ey_min, ex_max, ey_max) in boites_existantes:
        if not (nx_max + marge < ex_min or nx_min - marge > ex_max or ny_max + marge < ey_min or ny_min - marge > ey_max):
            return True 
    return False

# ==========================================
# 4. GÉNÉRATION DES IMAGES ARTIFICIELLES
# ==========================================
print("\n2. Génération des images artificielles...")
fichiers_psos = [f for f in os.listdir(dossier_psos) if f.endswith(('.jpg', '.png'))]
fichiers_fonds = [f for f in os.listdir(dossier_fonds) if f.endswith(('.jpg', '.png', '.jpeg'))]

# On commence les nouveaux IDs à partir de 20000 pour éviter tout doublon
image_id_counter = 20000 
annotation_id_counter = 20000 

for i in range(NOMBRE_IMAGES_ARTIFICIELLES):
    fond_name = random.choice(fichiers_fonds)
    fond_path = os.path.join(dossier_fonds, fond_name)
    
    # Ouvrir le fond et le FORCER à la taille standard de la caméra
    img_fond = Image.open(fond_path).convert("RGB")
    img_fond = img_fond.resize((standard_w, standard_h), Image.Resampling.LANCZOS)
    
    nb_psos_a_coller = random.randint(2, 4) # On colle entre 2 et 4 PSOs
    boites_occupees = []
    
    for _ in range(nb_psos_a_coller):
        pso_name = random.choice(fichiers_psos)
        pso_path = os.path.join(dossier_psos, pso_name)
        img_pso = Image.open(pso_path).convert("RGB")
        
        largeur_pso, hauteur_pso = img_pso.size
        
        # Vérification si le PSO est trop grand
        if largeur_pso > (standard_w - 2*MARGE_BORD) or hauteur_pso > (standard_h - 2*MARGE_BORD):
            continue 

        place_trouvee = False
        essais = 0
        while not place_trouvee and essais < 50:
            # On restreint les coordonnées pour respecter la MARGE_BORD
            x_min = random.randint(MARGE_BORD, standard_w - largeur_pso - MARGE_BORD)
            y_min = random.randint(MARGE_BORD, standard_h - hauteur_pso - MARGE_BORD)
            
            nouvelle_boite = (x_min, y_min, x_min + largeur_pso, y_min + hauteur_pso)
            
            if not verifier_collision(nouvelle_boite, boites_occupees, marge=25):
                place_trouvee = True
                boites_occupees.append(nouvelle_boite)
            essais += 1
            
        if not place_trouvee:
            continue
        
        img_fond.paste(img_pso, (x_min, y_min))
        
        # Ajout de l'annotation
        data_final["annotations"].append({
            "id": annotation_id_counter,
            "image_id": image_id_counter,
            "category_id": 1,
            "bbox": [x_min, y_min, largeur_pso, hauteur_pso],
            "area": largeur_pso * hauteur_pso,
            "iscrowd": 0
        })
        annotation_id_counter += 1
        
    nom_image_generee = f"artificiel_{image_id_counter}.jpg"
    img_fond.save(os.path.join(dossier_sortie_images, nom_image_generee))
    
    data_final["images"].append({
        "id": image_id_counter,
        "file_name": nom_image_generee,
        "width": standard_w,
        "height": standard_h
    })
    
    image_id_counter += 1

# ==========================================
# 5. SAUVEGARDE DU DATASET FINAL
# ==========================================
with open(json_sortie, 'w') as f:
    json.dump(data_final, f, indent=4)

print("\n" + "="*50)
print(f" SUCCÈS ! Le dataset d'entraînement est prêt.")
print(f" -> {NOMBRE_IMAGES_ARTIFICIELLES} images artificielles générées.")
print(f" -> Les images (vraies + fausses) sont dans : {dossier_sortie_images}")
print(f" -> Le fichier JSON final est : {json_sortie}")
print("==================================================")