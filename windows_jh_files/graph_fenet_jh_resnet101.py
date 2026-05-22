# graph.py
import matplotlib.pyplot as plt
import numpy as np

def plot_and_save_losses(train_losses, val_losses, filename="resnet101_loss_curve_jh.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Val Loss", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Évolution de la Perte")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_precision_recall(map_dict, filename="resnet101_fenet_jh_precision_recall_jh.png"):
    metrics = ['mAP@50', 'mAP@75', 'Rappel@1', 'Rappel@10', 'Rappel@100']
    values = [
        map_dict['map_50'].item(), map_dict['map_75'].item(),
        map_dict['mar_1'].item(), map_dict['mar_10'].item(), map_dict['mar_100'].item()
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['skyblue', 'blue', 'orange', 'orangered', 'red'])
    plt.ylim(0, 1.1)
    plt.ylabel("Score (0 à 1)")
    plt.title("Métriques de Performance du Modèle: école jouhaux")
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center')

    plt.savefig(filename)
    plt.close()

def plot_threshold_analysis(thresholds, precisions, recalls, f1_scores, best_threshold, filename="analyse_seuils_fenet_jh.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label="Précision", color="blue", linewidth=2, marker='o')
    plt.plot(thresholds, recalls, label="Rappel", color="red", linewidth=2, marker='s')
    plt.plot(thresholds, f1_scores, label="F1-Score", color="green", linewidth=3, linestyle="--")
    plt.axvline(x=best_threshold, color='black', linestyle=':', label=f'Seuil Optimal ({best_threshold:.2f})')

    plt.xlabel("Seuil de Confiance")
    plt.ylabel("Score")
    plt.title("Performances selon le seuil de confiance: cas des fenêtre jouhaux")
    plt.legend(loc="lower center")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close() 