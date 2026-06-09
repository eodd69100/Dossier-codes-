import torch
import torch.nn.functional as F_nn
import cv2
from PIL import Image
from torchvision import transforms

from model_fenet_convnext import create_model

def predict_window_state(model, image_path, device):
    """ Prend UNE image de fenêtre découpée et prédit son état. """
    #  Charger l'image avec PIL
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Erreur : Impossible de trouver l'image {image_path}")
        return None, None

    # Préparer l'image EXACTEMENT comme ConvNeXt l'attend (224x224)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Ajouter la dimension "Batch" pour faire [1, 3, 224, 224]
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Prédiction
    with torch.no_grad():
        outputs = model(img_tensor)

    # Traduire la sortie mathématique en pourcentages avec Softmax
    probabilites = F_nn.softmax(outputs, dim=1)[0]
    
    # En PyTorch, les dossiers sont souvent lus par ordre alphabétique :
    # 0 = ferme, 1 = ouvert
    score_ferme = probabilites[0].item() * 100
    score_ouvert = probabilites[1].item() * 100

    #  Déterminer le gagnant
    if score_ouvert > score_ferme:
        return "Ouvert", score_ouvert
    else:
        return "Ferme", score_ferme


def main(model_path, image_test_path):
    print("="*50)
    print(" Test du modèle Expert ConvNeXt (Classification) ")
    print("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé : {device}")

    # Chargement du modèle (2 classes)
    print("\nChargement du modèle...")
    model = create_model(num_classes=2)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(" -> Modèle chargé avec succès !")
    except FileNotFoundError:
        print(f"Erreur : Le fichier {model_path} est introuvable.")
        return

    model = model.to(device)
    model.eval()

    # Prédiction sur l'image
    print("\nAnalyse de l'image en cours...")
    etat, confiance = predict_window_state(model, image_test_path, device)
    
    if etat is None:
        return

    # Affichage des résultats dans le terminal
    print("\n" + "="*40)
    print(" RÉSULTAT DE L'ANALYSE :")
    print(f" -> Cette fenêtre est {etat.upper()} (Certitude : {confiance:.1f}%)")
    print("="*40)

    # Sauvegarde visuelle avec OpenCV
    img_cv2 = cv2.imread(image_test_path)
    if img_cv2 is not None:
        # On écrit le texte sur l'image
        texte = f"Etat: {etat} ({confiance:.1f}%)"
        couleur = (0, 255, 0) if etat == "Ouvert" else (0, 0, 255) # Vert si ouvert, Rouge si fermé
        
        # Petit fond noir pour la lisibilité
        cv2.rectangle(img_cv2, (0, 0), (350, 40), (0, 0, 0), -1)
        cv2.putText(img_cv2, texte, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur, 2)
        
        nom_sortie = "resultat_classification_convnext.jpg"
        cv2.imwrite(nom_sortie, img_cv2)
        print(f"\nImage de résultat sauvegardée sous '{nom_sortie}'")


if __name__ == "__main__":
    # Tes chemins vers le modèle ConvNeXt et la petite image croppée
    model_path = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\windows_anatole_files\convnextv2_expert_fenetres.pth"
    image_test_path = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\brouillon\pso_img1_ann30.jpg"
    
    main(model_path, image_test_path)