# data.py
import os
import json
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader

class COCODatasetPSO(Dataset):
    def __init__(self, img_folder, annotation_file):
        self.img_folder = img_folder
        
        with open(annotation_file) as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]
        self.cat_to_label = {1: 1}  # 1 seule classe = PSO

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
        img_tensor = F.to_tensor(img)

        boxes, labels = [], []

        for ann in self.image_to_annots.get(img_id, []):
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloaders(train_img, train_ann, val_img, val_ann, batch_size=2):
    train_dataset = COCODatasetPSO(img_folder=train_img, annotation_file=train_ann)
    val_dataset = COCODatasetPSO(img_folder=val_img, annotation_file=val_ann)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader