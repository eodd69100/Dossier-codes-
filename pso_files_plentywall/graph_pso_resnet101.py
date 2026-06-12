import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision 
import torch

def plot_and_save_losses(train_losses, val_losses, train_cls_losses, train_reg_losses, filename="resnet101_loss_curve.png"):
    plt.figure(figsize=(12, 6))
    
    # Les pertes globales (Les lignes principales)
    plt.plot(train_losses, label="Train Loss (Totale)", marker="o", color="blue", linewidth=2)
    plt.plot(val_losses, label="Val Loss (Totale)", marker="s", color="orange", linewidth=2)
    
    # Le détail sous le capot (En pointillés pour ne pas surcharger)
    plt.plot(train_cls_losses, label="Train Loss (Classification - Le doute)", linestyle="--", color="green", alpha=0.7)
    plt.plot(train_reg_losses, label="Train Loss (Régression - La précision de la boîte)", linestyle=":", color="red", alpha=0.7)
    
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Évolution détaillée des Pertes (Totale, Classification, Régression)", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_map_iou_curve(iou_thresholds, map_values, global_map, filename="resnet101_map_iou_curve.png"):
    plt.figure(figsize=(10, 6))
    
    # Tracer la courbe de dégradation
    plt.plot(iou_thresholds, map_values, marker='o', color='blue', linewidth=2, label="mAP à chaque seuil IoU")
    
    # Remplir la zone sous la courbe.
    plt.fill_between(iou_thresholds, map_values, color='skyblue', alpha=0.3)
    
    # Tracer la ligne de la moyenne (le mAP global)
    plt.axhline(y=global_map, color='red', linestyle='--', linewidth=2, label=f"mAP Global (Moyenne) = {global_map:.4f}")

    plt.xlabel("Seuil d'exigence IoU (Intersection over Union)", fontsize=12, fontweight='bold')
    plt.ylabel("Score mAP (0 à 1)", fontsize=12, fontweight='bold')
    plt.title("Dégradation du mAP selon la sévérité de l'IoU: cas des fenêtres", fontsize=14, fontweight="bold")
    
    # Forcer l'affichage de tous les pas sur l'axe X
    plt.xticks(iou_thresholds)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Ajouter les valeurs exactes au-dessus de chaque point
    for i, txt in enumerate(map_values):
        plt.text(iou_thresholds[i], map_values[i] + 0.02, f"{txt:.2f}", ha='center', fontsize=9)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_threshold_analysis(thresholds, precisions, recalls, f1_scores, best_threshold, filename="analyse_seuils_fenetres.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label="Précision", color="blue", linewidth=2, marker='o')
    plt.plot(thresholds, recalls, label="Rappel", color="red", linewidth=2, marker='s')
    plt.plot(thresholds, f1_scores, label="F1-Score", color="green", linewidth=3, linestyle="--")
    plt.axvline(x=best_threshold, color='black', linestyle=':', label=f'Seuil Optimal ({best_threshold:.2f})')

    plt.xlabel("Seuil de Confiance")
    plt.ylabel("Score")
    plt.title("Performances selon le seuil de confiance")
    plt.legend(loc="lower center")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


