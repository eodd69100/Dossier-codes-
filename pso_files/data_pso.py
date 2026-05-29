# data.py
import os
import json
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors

class COCODatasetPSO(Dataset):
    """Dataset personnalisé pour le format COCO adapté à notre tâche de détection de pso.
      Il lit les images et leurs annotations à partir d'un dossier et d'un fichier JSON,"""
    def __init__(self, img_folder, annotation_file, transforms=None):
        self.img_folder = img_folder
        self.transforms = transforms # On stocke nos transformations
        
        with open(annotation_file) as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]
        self.cat_to_label = {1: 1,2:1}  # pso a l'id 1 et 2 dans les annotations, et on veut que le modèle le reconnaisse comme la classe 1 (pas de classe 0 pour le fond)

        self.image_to_annots = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.image_to_annots:
                self.image_to_annots[img_id] = []
            self.image_to_annots[img_id].append(ann)

        self.id_to_filename = {img["id"]: img["file_name"] for img in self.images}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.images[idx]["id"]
        filename = self.id_to_filename[img_id]
        img_path = os.path.join(self.img_folder, filename)

        img = Image.open(img_path).convert("RGB")
        largeur, hauteur = img.size # On a besoin de la taille pour les boîtes

        boxes, labels = [], []

        for ann in self.image_to_annots.get(img_id, []):
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            
            cat_id = ann["category_id"]
            labels.append(self.cat_to_label[cat_id])  

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)

        # On emballe nos boîtes dans un objet "BoundingBoxes".
        # Ainsi, si v2 retourne l'image, il saura comment retourner ces boîtes !
        tv_boxes = tv_tensors.BoundingBoxes(
            boxes_tensor, 
            format="XYXY", 
            canvas_size=(hauteur, largeur)
        )

        target = {
            "boxes": tv_boxes,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id])
        }

        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# PIPELINE DE DATA AUGMENTATION 

def get_transforms(train):
    transforms = []
    
    # Transformations UNIQUEMENT pour l'entraînement
    if train:
        #  Symétrie horizontale (1 chance sur 2)
        # Ça casse la mémorisation du "les pso sont toujours à gauche"
        transforms.append(v2.RandomHorizontalFlip(p=0.5))
        
        # Jittering de couleur
        # Simule des photos prises le matin, le soir, sous les nuages ou en plein soleil
        transforms.append(v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)) #
        
        # Zoom / Dézoom aléatoire (Échelle)
        # Force le modèle à comprendre la texture d'un pso peu importe sa taille
        transforms.append(v2.RandomZoomOut(fill=0, p=0.2)) # 
        # rotation aléatoire (petits angles)
       # transforms.append(v2.RandomRotation(degrees=(-10, 10))) # on peut faire du -15 à +15 degrés, mais pas plus pour éviter de faire des images irréalistes
    # Transformations OBLIGATOIRES: (pour Train ET Validation)
    # Convertit l'image PIL en Tenseur PyTorch normalisé
    transforms.append(v2.ToImage()) 
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    
    return v2.Compose(transforms)

def get_dataloaders(train_img, train_ann, val_img, val_ann, batch_size=2): 
    # On applique les augmentations lourdes sur le Train
    train_dataset = COCODatasetPSO(
        img_folder=train_img, 
        annotation_file=train_ann, 
        transforms=get_transforms(train=True) 
    )
    
    # On n'applique AUCUNE augmentation sur la Validation (juste la conversion en Tenseur)
    val_dataset = COCODatasetPSO(
        img_folder=val_img, 
        annotation_file=val_ann, 
        transforms=get_transforms(train=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader