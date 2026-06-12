import json

def standardiser_coco(chemin_entree, chemin_sortie):
    print(f"Nettoyage de {chemin_entree}...")
    
    # 1. Chargement du fichier
    with open(chemin_entree, 'r') as f:
        data = json.load(f)
        
    # 2. Remplacement strict des catégories
    data["categories"] = [
        {
            "id": 1,
            "name": "pso",
            "supercategory": "none"
        }
    ]
    
    # 3. Forcer TOUTES les Bounding Boxes à cibler la catégorie 1
    annotations_modifiees = 0
    for ann in data["annotations"]:
        if ann["category_id"] != 1:
            ann["category_id"] = 1
            annotations_modifiees += 1
            
    # 4. Sauvegarde
    with open(chemin_sortie, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f" -> Succès ! {annotations_modifiees} annotations corrigées.")
    print(f" -> Fichier sauvegardé sous : {chemin_sortie}\n")


if __name__ == "__main__":
    # --- À MODIFIER AVEC TES CHEMINS ---
    
    # Nettoyage de la base TRAIN
   # train_in = r"C:\Chemin\Vers\Ton\train.json"
    #train_out = r"C:\Chemin\Vers\Ton\train_propre.coco.json"
    #standardiser_coco(train_in, train_out)
    
    # Nettoyage de la base VAL
    val_in = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\train_artificiel.json"
    val_out = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\val_propre.coco.json"
    standardiser_coco(val_in, val_out)