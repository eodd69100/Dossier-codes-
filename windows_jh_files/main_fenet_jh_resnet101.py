
import torch
import numpy as np

# Import depuis nos propres fichiers
from Codes.windows_jh_files.data_fenet_jh import get_dataloaders
from Codes.windows_jh_files.model_fenet_jh_resnet101 import create_model
from Codes.windows_jh_files.engine_fenet_jh_resnet101 import train_one_epoch, evaluate_loss, evaluate_sweet_spot,evaluate_map_all_ious
from Codes.windows_jh_files.graph_fenet_jh_resnet101 import plot_and_save_losses, plot_threshold_analysis



def main(train_img,train_ann,val_img,val_ann ):
    print("="*40)
    print("1. Configuration des données")
    print("="*40)

    train_loader, val_loader = get_dataloaders(train_img, train_ann, val_img, val_ann, batch_size=1)

    print("="*40)
    print("Initialisation du Modèle")
    print("="*40)
    num_classes=3 # Background (0) + Fermé (1) + Ouvert (2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé : {device}")
    
    model = create_model(num_classes=num_classes, pretrained=True).to(device)

    print("="*40)
    print("Phase d'Optimisation")
    print("="*40)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0007)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    num_epochs = 100 
    train_losses, val_losses = [], []
    best_val_loss = float('inf')  
    patience_early_stop = 7 
    trigger_times = 0 

    print("\n--- Début de l'entraînement ---")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss = evaluate_loss(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[Epoch {epoch+1}/{num_epochs}] LR={current_lr:.6f} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

        # EARLY STOPPING
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), "faster_resnet101_fenet_jh_best.pth")
            print("  -> Nouveau meilleur modèle sauvegardé")
        else:
            trigger_times += 1
            print(f"  -> Early Stop patience : {trigger_times}/{patience_early_stop}")
            if trigger_times >= patience_early_stop:
                print("Early stopping déclenché")
                break

    print("\n="*40)
    print("Évaluation et Graphiques")
    print("="*40)

    plot_and_save_losses(train_losses, val_losses)
    print("Graphique de perte sauvegardé.")

    # Recharger le meilleur modèle pour l'évaluation finale
    model.load_state_dict(torch.load("faster_resnet101_fenet_jh_best.pth"))
    

    print("\nCalcul de la mAP sur le jeu de validation...")
    
    map_dict, iou_threshs, map_vals = evaluate_map_all_ious(model, val_loader, device)
    
    print(f"mAP (IoU=0.50)    : {map_dict['map_50'].item():.4f}")
    print(f"mAP (IoU=0.75)    : {map_dict['map_75'].item():.4f}")
    print(f"mAP GLOBAL        : {map_dict['map'].item():.4f}") # La moyenne !
    
    # On génère le nouveau graphique
    from Codes.windows_jh_files.graph_fenet_jh_resnet101 import plot_map_iou_curve
    plot_map_iou_curve(iou_threshs, map_vals, map_dict['map'].item(), "resnet101_fenet_jh_map_iou_curve.png")
    print("Graphique de l'évolution du mAP sauvegardé.")

    print("\nRecherche du seuil de confiance optimal...")
    thresholds, precisions, recalls, f1_scores = evaluate_sweet_spot(model, val_loader, device)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"--- RÉSULTAT OPTIMAL ---")
    print(f"Meilleur seuil : {best_threshold:.2f} (Précision: {precisions[best_idx]*100:.1f}%, Rappel: {recalls[best_idx]*100:.1f}%)")
    
    plot_threshold_analysis(thresholds, precisions, recalls, f1_scores, best_threshold)
    print("Graphique des seuils sauvegardé.")

if __name__ == "__main__":
    train_img = r"C:\Users\k.nguessan\Desktop\DossierStage\Continuous_PSO.v6i.coco\train\images"
    train_ann = r"C:\Users\k.nguessan\Desktop\DossierStage\Continuous_PSO.v6i.coco\train\_annotations.coco.json"
    val_img = r"C:\Users\k.nguessan\Desktop\DossierStage\Continuous_PSO.v6i.coco\valid\images"
    val_ann = r"C:\Users\k.nguessan\Desktop\DossierStage\Continuous_PSO.v6i.coco\valid\_annotations.coco.json"
    main(train_img,train_ann,val_img,val_ann ) 
