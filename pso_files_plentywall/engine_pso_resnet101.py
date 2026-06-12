import torch
import numpy as np
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def train_one_epoch(model, optimizer, data_loader, device):
    """Effectue une époque d'entraînement en suivant séparément les pertes de classification 
    et de régression des boîtes, avec sécurité anti-explosion de gradient."""
    model.train()
    
    running_total_loss = 0.0
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Le modèle calcule les composants de la perte
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Extraction des pertes individuelles pour le suivi détaillé
        perte_totale_batch = losses.item()
        perte_cls_batch = loss_dict['loss_classifier'].item()
        perte_reg_batch = loss_dict['loss_box_reg'].item()
        
        # Accumulation dans nos compteurs
        running_total_loss += perte_totale_batch
        running_cls_loss += perte_cls_batch
        running_reg_loss += perte_reg_batch
        
        # Rétropropagation
        optimizer.zero_grad()
        losses.backward()
        
        # SÉCURITÉ : Empêche les gradients d'exploser en les limitant à une norme maximale de 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
    # Calcul de la moyenne par batch pour cette époque
    num_batches = len(data_loader)
    epoch_total_loss = running_total_loss / num_batches
    epoch_cls_loss = running_cls_loss / num_batches
    epoch_reg_loss = running_reg_loss / num_batches
    
    return epoch_total_loss, epoch_cls_loss, epoch_reg_loss


def evaluate_loss(model, data_loader, device):
    """Calcule la perte moyenne sur le jeu de validation.
    Note : Faster R-CNN requiert model.train() pour accepter de calculer une perte."""
    model.train()  # Crucial pour forcer le calcul des pertes de validation
    running_val_loss = 0.0
    
    with torch.no_grad():  # On désactive la mise à jour des poids
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            running_val_loss += losses.item()
            
    return running_val_loss / len(data_loader)


def evaluate_map_all_ious(model, val_loader, device):
    """Évalue le mAP du modèle sur l'ensemble de validation pour une gamme de seuils d'IoU (0.50 à 0.95)."""
    model.eval() # Mode évaluation standard pour obtenir des boîtes prédictives
    
    all_preds = []
    all_targets = []
    
    print("Extraction des prédictions pour le calcul mAP...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            preds = model(images)
            
            preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
            
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    metric_global = MeanAveragePrecision()
    metric_global.update(all_preds, all_targets)
    map_dict = metric_global.compute()
    
    iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    map_values = []
    
    print("Calcul détaillé par seuil d'IoU...")
    for iou in iou_thresholds:
        metric_temp = MeanAveragePrecision(iou_thresholds=[iou])
        metric_temp.update(all_preds, all_targets)
        temp_dict = metric_temp.compute()
        map_values.append(temp_dict['map'].item())
        
    return map_dict, iou_thresholds, map_values


def evaluate_sweet_spot(model, val_loader, device):
    """Calcule la précision, le rappel et le F1-score selon les seuils de confiance."""
    model.eval()
    y_true_raw, y_scores_raw = [], [] 
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