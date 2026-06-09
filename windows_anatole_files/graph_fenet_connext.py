import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ==========================================
# COURBES D'APPRENTISSAGE (LOSS)
# ==========================================
def generer_courbes(train_losses, val_losses, save_path="courbe_perte.png"):
    """ Génère la courbe de chute de la perte (Loss) """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Perte Entraînement", color="#1f77b4", linewidth=2)
    plt.plot(val_losses, label="Perte Validation", color="#ff7f0e", linewidth=2)
    
    plt.title("Évolution de la Perte (Loss) au fil des Époques", fontsize=14, pad=15)
    plt.xlabel("Époques", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Graphique des pertes sauvegardé sous : {save_path}")


# ==========================================
# MATRICE DE CONFUSION (HEATMAP) POUR CONVNEXT
# ==========================================
def generer_matrice_confusion(y_vraies, y_predites, classes=["Fermé", "Ouvert"], save_path="heatmap_confusion.png"):
    """ Génère une Heatmap (Matrice de Confusion) pour visualiser les erreurs de classification """
    cm = confusion_matrix(y_vraies, y_predites)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 16})
    
    plt.title("Matrice de Confusion - État des Fenêtres", fontsize=14, pad=15)
    plt.xlabel("Prédictions du Modèle", fontsize=12)
    plt.ylabel("Réalité (Vérité Terrain)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"🔥 Heatmap de confusion sauvegardée sous : {save_path}")


# ==========================================
# COURBE PRECISION-RECALL (PR CURVE) POUR CONVNEXT
# ==========================================
def generer_courbe_pr(y_vraies, y_scores, save_path="courbe_pr.png"):
    """ 
    Génère la courbe Précision-Rappel.
    y_scores : Les probabilités brutes (ex: probabilité que ce soit "Ouvert")
    """
    precision, recall, seuils = precision_recall_curve(y_vraies, y_scores)
    ap = average_precision_score(y_vraies, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', linewidth=2, label=f'AP = {ap:.2f}')
    
    # Remplissage sous la courbe pour le style
    plt.fill_between(recall, precision, alpha=0.2, color='purple')

    plt.title("Courbe Précision-Rappel (Classe 'Ouvert')", fontsize=14, pad=15)
    plt.xlabel("Rappel (Recall)", fontsize=12)
    plt.ylabel("Précision (Precision)", fontsize=12)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Courbe Precision-Recall sauvegardée sous : {save_path}")



