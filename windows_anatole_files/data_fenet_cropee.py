import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

"""
Ce script implémente un dataset personnalisé pour la classification des fenêtres en "fermé" ou "ouvert" à partir d'une arborescence de dossiers.
Il inclut des fonctions de prétraitement spécifiques pour rendre les images carrées sans déformation, ainsi qu'une égalisation d'histogramme locale (CLAHE) pour améliorer le contraste et faire ressortir les montants et perspectives des fenêtres.   


"""

class WindowFolderDataset(Dataset):
    def __init__(self, dossier_racine, split="train", validation_split=0.2, seed=42, target_size=224):
        self.target_size = target_size
        self.class_to_idx = {"ferme": 0, "ouvert": 1} 
        
        # Collecte de toutes les images présentes dans l'arborescence
        toutes_les_images = []
        for nom_classe, idx in self.class_to_idx.items():
            chemin_dossier = os.path.join(dossier_racine, nom_classe)
            if not os.path.exists(chemin_dossier):
                continue
            for nom_img in os.listdir(chemin_dossier):
                if nom_img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    toutes_les_images.append({
                        "chemin": os.path.join(chemin_dossier, nom_img),
                        "label": idx
                    })
        
        # Mélange reproductible (Seed) pour isoler le Train et la Validation
        toutes_les_images.sort(key=lambda x: x["chemin"])
        random.seed(seed)
        random.shuffle(toutes_les_images)
        
        # Calcul de la découpe (ex: 80% Train / 20% Val)
        total_images = len(toutes_les_images)
        index_coupure = int(total_images * (1 - validation_split))
        
        if split == "train":
            self.liste_images = toutes_les_images[:index_coupure]
            # Data Augmentation pour l'entraînement !
            self.transform = transforms.Compose([
                transforms.ToPILImage(),            # Convertit le tableau NumPy en image PIL pour les transformations
                transforms.RandomHorizontalFlip(p=0.5), # Effet miroir
                transforms.RandomRotation(degrees=10),  # Légère inclinaison
                transforms.ColorJitter(brightness=0.2, contrast=0.2), # Variation de lumière
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalisation standard pour les modèles pré-entraînés
            ])
        else:
            self.liste_images = toutes_les_images[index_coupure:]
            # Normalisation stricte pour la validation (On ne modifie pas les images de test)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        print(f"📊 Mode [{split.upper()}] : {len(self.liste_images)} images chargées.")

    def __len__(self):
        return len(self.liste_images)

    def appliquer_padding_carre(self, crop_img):
        """Ajoute des bandes noires pour rendre l'image carrée sans déformer la fenêtre."""
        h, w = crop_img.shape[:2] # On récupère la hauteur et la largeur du crop
        max_side = max(h, w)
        toile_carree = np.zeros((max_side, max_side, 3), dtype=np.uint8) # Image noire de taille max_side x max_side
        x_offset = (max_side - w) // 2 # Calcul du décalage horizontal pour centrer le crop
        y_offset = (max_side - h) // 2 # Calcul du décalage vertical pour centrer le crop
        toile_carree[y_offset:y_offset+h, x_offset:x_offset+w] = crop_img # On place le crop au centre de la toile carrée
        return cv2.resize(toile_carree, (self.target_size, self.target_size)) 

    def appliquer_clahe(self, img):
        """Égalise l'histogramme local pour faire ressortir les montants et perspectives."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # On convertit l'image en espace de couleur LAB pour travailler sur la luminosité
        l, a, b = cv2.split(lab) # On sépare les canaux L (luminosité), A et B (couleurs)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4 )) # Création de l'objet CLAHE avec une limite de contraste et une taille de grille
        cl = clahe.apply(l) # Application de CLAHE sur le canal de luminosité pour améliorer les détails locaux
        img_fusion = cv2.merge((cl, a, b)) # On fusionne à nouveau les canaux pour obtenir l'image finale avec un meilleur contraste
        return cv2.cvtColor(img_fusion, cv2.COLOR_LAB2BGR)

    def __getitem__(self, idx):
        info = self.liste_images[idx]
        img_bgr = cv2.imread(info["chemin"])
        
        if img_bgr is None:
            img_bgr = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        # Application de notre pipeline de prétraitement morphologique
        img_carre = self.appliquer_padding_carre(img_bgr)
        img_contrast = self.appliquer_clahe(img_carre)
        
        # Passage en RGB
        img_rgb = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2RGB)
        
        # Application des transformations PyTorch (Augmentation + Tensor + Normalize)
        img_tensor = self.transform(img_rgb)
        
        return img_tensor, info["label"]


def get_dataloaders(dossier_unique, batch_size=16, val_split=0.2):
    # Initialisation des datasets
    train_dataset = WindowFolderDataset(dossier_unique, split="train", validation_split=val_split)
    val_dataset = WindowFolderDataset(dossier_unique, split="val", validation_split=val_split)

    
    
    # Extraire tous les labels de l'entraînement
    labels_train = [info["label"] for info in train_dataset.liste_images]
    
    #Compter combien on a de chaque classe
    compte_classes = np.bincount(labels_train) # compte_classes[0] = nombre d'images "ferme", compte_classes[1] = nombre d'images "ouvert"
    print(f"Équilibre du Train : Fermé={compte_classes[0]}, Ouvert={compte_classes[1]}")
    
    # Calculer le poids de la classe (plus la classe est rare, plus le poids est fort)
    # On utilise un float pour éviter les divisions par zéro
    poids_par_classe = 1.0 / np.maximum(compte_classes, 1.0)
    
    #  Attribuer ce poids à CHAQUE image selon son label
    poids_echantillons = [poids_par_classe[label] for label in labels_train]
    
    #  Créer l'objet Sampler
    sampler = WeightedRandomSampler(
        weights=poids_echantillons,
        num_samples=len(poids_echantillons), # Il tirera autant d'images qu'une époque normale
        replacement=True # Autorise à tirer la même image rare plusieurs fois
    )
    # Création des DataLoaders
    # Comme on utilise un 'sampler', on ne peut PAS utiliser 'shuffle=True'.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler, #
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, val_loader




if __name__ == "__main__":
    DOSSIER_UNIQUE_CLASSIF = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\dataset_classification"

    get_dataloaders(DOSSIER_UNIQUE_CLASSIF, batch_size=4, val_split=0.2) 

    im_chemin = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\dataset_classification\ouvert\pso_img4_ann88.jpg"
    
    # Charger l'image
    img_bgr = cv2.imread(im_chemin)
    
    if img_bgr is None:
        print("Erreur : Image introuvable.")
    else:
        # 2. On crée une instance de classe bidon juste pour accéder à la méthode 
        # (car appliquer_clahe est dans la classe)
        # OU alors, déplace la fonction en dehors de la classe si tu veux l'appeler simplement
        
        # Astuce rapide : on instancie le dataset juste pour le test
        ds = WindowFolderDataset(dossier_racine=r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\dataset_classification")
        
        # 3. Test de CLAHE
        img_clahe = ds.appliquer_clahe(img_bgr)
        
        # Affichage et sauvegarde pour comparer
        #cv2.imshow("Originale", img_bgr)
        #cv2.imshow("Apres CLAHE", img_clahe)
        cv2.imwrite("test_clahe.jpg", img_clahe)
        cv2.imwrite("test_originale.jpg", img_bgr)
        print("Appuie sur n'importe quelle touche pour fermer les fenêtres...")
        cv2.waitKey(0)
        cv2.destroyAllWindows() 