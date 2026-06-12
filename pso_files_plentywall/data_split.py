import json
import random
import os

print("="*40)
print(" Séparation du Dataset (50% Réel / 50% Donneurs)")
print("="*40)

# --- 1. PARAMÈTRES ---
chemin_json_original = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Continuous_PSO.v6i.coco\train\_annotations.coco.json"

dossier_base = os.path.dirname(chemin_json_original)
json_a_garder = os.path.join(dossier_base, "train_a_garder.json")
json_donneurs = os.path.join(dossier_base, "train_donneurs.json")

# Ratio de séparation (0.5 = 50% gardés, 50% découpés)
RATIO = 0.5 

# --- 2. LECTURE DU JSON ---
with open(chemin_json_original, 'r') as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']
categories = data['categories']

# --- 3. MÉLANGE ET SÉPARATION ---
# On fixe le seed pour que le hasard soit toujours le même si tu relances le script
random.seed(42) 
random.shuffle(images)

index_coupure = int(len(images) * RATIO)

images_gardees = images[:index_coupure]
images_donneuses = images[index_coupure:]

# Création de listes d'IDs pour trier facilement les annotations
ids_gardes = {img['id'] for img in images_gardees}
ids_donneurs = {img['id'] for img in images_donneuses}

# On sépare les annotations correspondantes
ann_gardees = [ann for ann in annotations if ann['image_id'] in ids_gardes]
ann_donneuses = [ann for ann in annotations if ann['image_id'] in ids_donneurs]

# --- 4. CRÉATION DES NOUVEAUX DICTIONNAIRES ---
data_a_garder = {
    "images": images_gardees,
    "annotations": ann_gardees,
    "categories": categories
}

data_donneurs = {
    "images": images_donneuses,
    "annotations": ann_donneuses,
    "categories": categories
}

# --- 5. SAUVEGARDE ---
with open(json_a_garder, 'w') as f:
    json.dump(data_a_garder, f, indent=4)

with open(json_donneurs, 'w') as f:
    json.dump(data_donneurs, f, indent=4)

print(f"Total d'images original : {len(images)}")
print(f" -> Images conservées (Vraies façades) : {len(images_gardees)} (sauvegardé dans train_a_garder.json)")
print(f" -> Images sacrifiées (Pour découpe PSO) : {len(images_donneuses)} (sauvegardé dans train_donneurs.json)")
print("Terminé !")