import torch
import numpy as np

# Import depuis nos propres fichiers
from data_pso import get_dataloaders
from model_pso_resnet101 import create_model
from engine_pso_resnet101 import train_one_epoch, evaluate_loss, evaluate_sweet_spot, evaluate_map_all_ious
from graph_pso_resnet101 import plot_and_save_losses, plot_threshold_analysis, plot_map_iou_curve

"""
Ce script implémente une fonction d'inférence pour notre modèle de détection de PSO basé sur Faster R-CNN avec un backbone ResNet-101. 
Il charge le modèle entraîné, effectue la détection sur une image de test,  et affiche les résultats en annotant l'image avec les boîtes de détection et les pourcentages d'ouverture calculés à partir des hauteurs des boîtes détectées.
Il inclut également une fonction pour séparer les boîtes détectées en deux étages (haut et bas) selon leur position verticale, ainsi qu'une fonction de calibration pour ajuster les pourcentages d'ouverture en fonction de la hauteur des boîtes détectées.   """



def main(train_img, train_ann, val_img, val_ann):
    print("="*40)
    print("1. Configuration des données")
    print("="*40)

    train_loader, val_loader = get_dataloaders(train_img, train_ann, val_img, val_ann, batch_size=1)

    print("="*40)
    print("2. Initialisation du Modèle")
    print("="*40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé : {device}")
    num_classes = 2  ## background(0) + pso(1)
    model = create_model(num_classes=num_classes, pretrained=True).to(device)

    print("="*40)
    print("3. Phase d'Optimisation et Entraînement")
    print("="*40)

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0007)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0000525, weight_decay=0.009)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) # mode='min' pour surveiller la perte de validation

    num_epochs = 100
    best_val_loss = float('inf')  
    patience_early_stop = 10
    trigger_times = 0 

    print("\n--- Début de l'entraînement ---")
    historique_perte_totale = []
    historique_perte_classification = []
    historique_perte_regression = []
    val_losses = []

    for epoch in range(num_epochs):
        # 1. Entraînement et récupération des 3 pertes
        loss_totale, loss_cls, loss_reg = train_one_epoch(model, optimizer, train_loader, device)

        # 2. Sauvegarde dans les listes
        historique_perte_totale.append(loss_totale)
        historique_perte_classification.append(loss_cls)
        historique_perte_regression.append(loss_reg)
        
        # 3. Évaluation sur la validation
        val_loss = evaluate_loss(model, val_loader, device)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[Epoch {epoch+1}/{num_epochs}] LR={current_lr:.6f} | Train Loss={loss_totale:.4f} | Val Loss={val_loss:.4f}")

        # EARLY STOPPING
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), "faster_pso_resnet101_best.pth")
            print("  -> Nouveau meilleur modèle sauvegardé")
        else:
            trigger_times += 1
            print(f"  -> Early Stop patience : {trigger_times}/{patience_early_stop}")
            if trigger_times >= patience_early_stop:
                print("Early stopping déclenché !")
                break

    print("\n="*40)
    print("4. Évaluation et Graphiques")
    print("="*40)

    # Sauvegarde du graphique détaillé des pertes
    plot_and_save_losses(
        historique_perte_totale, 
        val_losses, 
        historique_perte_classification, 
        historique_perte_regression
    )
    print("Graphique de perte détaillé sauvegardé.")

    # Recharger les meilleurs poids pour l'évaluation finale
    print("\nChargement des meilleurs poids du modèle...")
    model.load_state_dict(torch.load("faster_pso_resnet101_best.pth"))
    
    # --- BLOC DE CALCUL DES MAP RÉINTRODUIT ---
    print("\nCalcul de la mAP sur le jeu de validation...")
    map_dict, iou_threshs, map_vals = evaluate_map_all_ious(model, val_loader, device)
    
    print(f"mAP (IoU=0.50)    : {map_dict['map_50'].item():.4f}")
    print(f"mAP (IoU=0.75)    : {map_dict['map_75'].item():.4f}")
    print(f"mAP GLOBAL        : {map_dict['map'].item():.4f}")
    
    plot_map_iou_curve(iou_threshs, map_vals, map_dict['map'].item(), "resnet101_map_iou_curve.png")
    print("Graphique de l'évolution du mAP sauvegardé.")

    # BLOC DE RECHERCHE DU SWEET SPOT SUR LA BASE DE VALIDATION
    print("\nRecherche du seuil de confiance optimal...")
    thresholds, precisions, recalls, f1_scores = evaluate_sweet_spot(model, val_loader, device)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"--- RÉSULTAT OPTIMAL ---")
    print(f"Meilleur seuil : {best_threshold:.2f} (Précision: {precisions[best_idx]*100:.1f}%, Rappel: {recalls[best_idx]*100:.1f}%)")
    
    plot_threshold_analysis(thresholds, precisions, recalls, f1_scores, best_threshold)
    print("Graphique d'analyse des seuils sur la base de validation sauvegardé.")

if __name__ == "__main__":
    train_img = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\images_artificielles"
    train_ann = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\train_artificiel.json"
    val_img = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\dataset_anatole_pso_2026.v1i.coco\valid\images"
    val_ann = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\dataset_anatole_pso_2026.v1i.coco\valid\_annotations.coco.json"
    main(train_img, train_ann, val_img, val_ann)
