import json
import numpy as np
from sklearn.cluster import KMeans

def calculer_ancres_optimales(json_path, nb_tailles=5, nb_ratios=6):
    """"""
    print("Analyse de vos PSO en cours...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    tailles = []
    ratios = []

    for ann in data['annotations']:
        # COCO donne les boîtes sous la forme [x, y, largeur, hauteur]
        w = ann['bbox'][2]
        h = ann['bbox'][3]

        if w == 0 or h == 0:
            continue

        # Dans PyTorch, la taille de l'ancre est la racine carrée de l'aire
        taille = np.sqrt(w * h)
        # Dans PyTorch, le ratio est Hauteur / Largeur
        ratio = h / w  

        tailles.append(taille)
        ratios.append(ratio)

    tailles = np.array(tailles).reshape(-1, 1)
    ratios = np.array(ratios).reshape(-1, 1)

    # 1. K-Means pour trouver les 5 tailles idéales (pour votre FPN à 5 niveaux)
    kmeans_tailles = KMeans(n_clusters=nb_tailles, random_state=42, n_init="auto").fit(tailles)
    tailles_optimales = np.sort(kmeans_tailles.cluster_centers_.flatten())

    # 2. K-Means pour trouver les proportions idéales (ex: très allongé, carré, etc.)
    kmeans_ratios = KMeans(n_clusters=nb_ratios, random_state=42, n_init="auto").fit(ratios)
    ratios_optimaux = np.sort(kmeans_ratios.cluster_centers_.flatten())

    # Formatage pour PyTorch
    tailles_tuple = tuple((round(t, 1),) for t in tailles_optimales)
    ratios_tuple = tuple(round(r, 2) for r in ratios_optimaux)

    print("\n" + "="*50)
    print("✅ RÉSULTAT : COPIEZ-COLLEZ CECI DANS model_pso_resnet101.py")
    print("="*50)
    print(f"    anchor_sizes = {tailles_tuple}")
    print(f"    aspect_ratios = ({ratios_tuple},) * len(anchor_sizes)")
    print("="*50 + "\n")

if __name__ == "__main__":
    #  chemin vers le JSON jeu d'entraînement (Train)
    chemin_train_json = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\dataset_anatole_pso_2026.v1i.coco\train\_annotations.coco.json"
    
    calculer_ancres_optimales(chemin_train_json)