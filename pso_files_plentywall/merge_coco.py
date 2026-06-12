import os
import json
import shutil

print("="*50)
print(" FUSION DE DATASETS COCO (Hard Negative Mining)")
print("="*50)

# ==========================================
# 1. PARAMÈTRES (À modifier avec tes chemins)
# ==========================================
# --- Ta base principale actuelle (celle qui marche bien) ---
dossier_images_principal = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\images_artificielles"
json_principal = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\train_artificiel.json"

# --- Tes 10 nouvelles images fraîchement annotées ---
dossier_nouvelles_images = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\add_data.coco\train\images"
json_nouveau = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\add_data.coco\train\_annotations.coco.json"

# ==========================================
# 2. CHARGEMENT DES DONNÉES
# ==========================================
with open(json_principal, 'r') as f:
    data_principal = json.load(f)

with open(json_nouveau, 'r') as f:
    data_nouveau = json.load(f)

# ==========================================
# 3. RECHERCHE DES IDS MAXIMUMS
# ==========================================
# On trouve l'ID le plus élevé dans la base principale pour ne rien écraser
max_img_id = max([img['id'] for img in data_principal['images']], default=0)
max_ann_id = max([ann['id'] for ann in data_principal['annotations']], default=0)

print(f"Base principale : {len(data_principal['images'])} images, {len(data_principal['annotations'])} annotations.")
print(f"Nouvelle base   : {len(data_nouveau['images'])} images, {len(data_nouveau['annotations'])} annotations.")

# ==========================================
# 4. FUSION DES IMAGES ET ANNOTATIONS
# ==========================================
# Ce dictionnaire va mémoriser l'ancien ID pour l'associer au nouvel ID décalé
mapping_img_id = {}

# --- A. Traitement des images ---
images_ajoutees = 0
for img in data_nouveau['images']:
    ancien_id = img['id']
    nom_fichier = img['file_name']
    
    # Création du nouvel ID sécurisé
    nouvel_id = max_img_id + 1 + images_ajoutees
    mapping_img_id[ancien_id] = nouvel_id
    
    # Mise à jour de l'ID dans le dictionnaire COCO
    img['id'] = nouvel_id
    data_principal['images'].append(img)
    
    # Copie physique de l'image vers le dossier principal
    src_path = os.path.join(dossier_nouvelles_images, nom_fichier)
    dst_path = os.path.join(dossier_images_principal, nom_fichier)
    
    if os.path.exists(src_path):
        if not os.path.exists(dst_path):
            shutil.copy(src_path, dst_path)
    else:
        print(f"[ATTENTION] Image introuvable : {src_path}")
        
    images_ajoutees += 1

# --- B. Traitement des annotations ---
annotations_ajoutees = 0
for ann in data_nouveau['annotations']:
    # On vérifie que l'annotation correspond bien à une image qu'on a ajoutée
    if ann['image_id'] in mapping_img_id:
        # On met à jour l'ID de l'image cible
        ann['image_id'] = mapping_img_id[ann['image_id']]
        
        # On met à jour l'ID propre de l'annotation
        ann['id'] = max_ann_id + 1 + annotations_ajoutees
        
        # On force la catégorie à être la même (généralement 1 pour PSO)
        ann['category_id'] = 1 
        
        data_principal['annotations'].append(ann)
        annotations_ajoutees += 1

# ==========================================
# 5. SAUVEGARDE
# ==========================================
with open(json_principal, 'w') as f:
    json.dump(data_principal, f, indent=4)

print("\n" + "="*50)
print(f" SUCCÈS DE LA FUSION !")
print(f" -> {images_ajoutees} images copiées dans : {dossier_images_principal}")
print(f" -> {annotations_ajoutees} annotations ajoutées dans : {json_principal}")
print(f" Le dataset principal est prêt pour le Fine-Tuning.")
print("==================================================")