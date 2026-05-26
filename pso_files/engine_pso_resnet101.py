
import torch
import numpy as np
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def train_one_epoch(model, optimizer, data_loader, device):
    """Effectue une époque d'entraînement sur le modèle Faster R-CNN.
    Args:      
        model: Le modèle Faster R-CNN à entraîner.
        optimizer: L'optimiseur utilisé pour mettre à jour les poids du modèle.
        data_loader: Le chargeur de données pour l'entraînement.
        device: Le dispositif (CPU ou GPU) sur lequel effectuer les calculs."""
    model.train() # Met le modèle en mode entraînement (nécessaire pour calculer les pertes sur Faster R-CNN)   
    train_loss_epoch = 0.0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets) # Le modèle retourne un dictionnaire de pertes (ex: {'loss_classifier': ..., 'loss_box_reg': ..., ...})
        loss = sum(loss_dict.values()) # On somme toutes les composantes de la perte pour obtenir une valeur scalaire totale

        optimizer.zero_grad() # Réinitialise les gradients accumulés des itérations précédentes
        loss.backward() # Effectue la rétropropagation pour calculer les gradients des poids du modèle par rapport à la perte
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Optionnel : clip les gradients pour éviter les explosions de gradients (utile surtout pour les réseaux profonds)
        optimizer.step() # Met à jour les poids du modèle en fonction des gradients calculés

        train_loss_epoch += loss.item() # Accumule la perte totale de l'époque pour pouvoir calculer la moyenne à la fin
    
    return train_loss_epoch / len(data_loader) # Retourne la perte moyenne par lot pour l'époque d'entraînement

def evaluate_loss(model, data_loader, device):
    """Évalue la perte du modèle sur un ensemble de validation."""

    model.train() # Nécessaire pour obtenir la validation loss sur Faster R-CNN
    val_loss_epoch = 0.0
    with torch.no_grad(): # On désactive le calcul des gradients pour économiser de la mémoire et accélérer les calculs pendant l'évaluation
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            val_loss_epoch += loss.item()

    return val_loss_epoch / len(data_loader)

def evaluate_map_all_ious(model, val_loader, device):
    """Évalue le mAP du modèle sur l'ensemble de validation pour une gamme de seuils d'IoU (de 0.50 à 0.95
par exemple)."""
    model.eval() # Met le modèle en mode évaluation (désactive les comportements spécifiques à l'entraînement comme le dropout)
    
    # On stocke tout sur le CPU pour ne faire l'inférence qu'une seule fois
    all_preds = []
    all_targets = []
    
    print("Extraction des prédictions pour le calcul mAP...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            preds = model(images)
            
            preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
            
            all_preds.extend(preds) # On ajoute les prédictions de ce lot à la liste globale de toutes les prédictions
            all_targets.extend(targets) # On ajoute les cibles de ce lot à la liste globale de toutes les cibles
            
    # Calcul du résumé global (pour avoir le vrai mAP moyen)
    metric_global = MeanAveragePrecision() # Par défaut, ce calcul se fait sur tous les seuils d'IoU de 0.50 à 0.95 avec un pas de 0.05
    metric_global.update(all_preds, all_targets) # On met à jour la métrique avec toutes les prédictions et cibles collectées
    map_dict = metric_global.compute()
    
    # Calcul détaillé pour notre courbe (0.50 à 0.95)
    iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    map_values = []
    
    print("Calcul détaillé par seuil d'IoU...")
    for iou in iou_thresholds:
        # On force la métrique à ne calculer que pour Ce seuil précis
        metric_temp = MeanAveragePrecision(iou_thresholds=[iou])
        metric_temp.update(all_preds, all_targets)
        temp_dict = metric_temp.compute()
        map_values.append(temp_dict['map'].item())
        
    return map_dict, iou_thresholds, map_values

def evaluate_sweet_spot(model, val_loader, device):
    """Évalue la précision, le rappel et le F1-score du modèle sur l'ensemble de validation pour une gamme de seuils de score de confiance 
        (de 0.0 à 0.95 par exemple) afin d'identifier le "sweet spot" optimal."""
    model.eval()
    y_true_raw, y_scores_raw = [], [] # Ces listes vont stocker les étiquettes de vérité terrain (1 pour vrai positif, 0 pour faux positif) 
                                       # et les scores de confiance correspondants pour toutes les prédictions à travers tous les seuils d'IoU
    total_gt_objects = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            preds = model(images)

            for i in range(len(preds)):
                pred_boxes = preds[i]['boxes'].cpu()
                pred_scores = preds[i]['scores'].cpu()
                gt_boxes = targets[i]['boxes'].cpu()

                total_gt_objects += len(gt_boxes)
                if len(pred_boxes) == 0: continue
                if len(gt_boxes) == 0:
                    y_true_raw.extend([0] * len(pred_scores))
                    y_scores_raw.extend(pred_scores.tolist())
                    continue

                indices_tries = torch.argsort(pred_scores, descending=True)
                pred_boxes = pred_boxes[indices_tries]
                pred_scores = pred_scores[indices_tries]

                ious = box_iou(pred_boxes, gt_boxes)
                gt_verrouilles = set() 

                for p_idx in range(len(pred_boxes)):
                    meilleur_iou = 0
                    meilleur_gt_idx = -1
                    for g_idx in range(len(gt_boxes)):
                        if ious[p_idx, g_idx] > meilleur_iou:
                            meilleur_iou = ious[p_idx, g_idx]
                            meilleur_gt_idx = g_idx
                    
                    if meilleur_iou >= 0.50 and meilleur_gt_idx not in gt_verrouilles:
                        y_true_raw.append(1) 
                        gt_verrouilles.add(meilleur_gt_idx) 
                    else:
                        y_true_raw.append(0) 
                    y_scores_raw.append(pred_scores[p_idx].item())

    # Analyse des seuils
    thresholds = np.arange(0.0, 0.95, 0.05)
    precisions, recalls, f1_scores = [], [], []

    for thresh in thresholds:
        valides = [i for i, score in enumerate(y_scores_raw) if score >= thresh]
        tp = sum(y_true_raw[i] for i in valides) 
        fp = len(valides) - tp                  
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_gt_objects if total_gt_objects > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return thresholds, precisions, recalls, f1_scores