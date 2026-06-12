import torch
import cv2
from PIL import Image
from torchvision.transforms import functional as F
from model_fenet_convnext import create_model

"""Ce script implémente une fonction d'extraction des coordonnées des fenêtres détectées par notre modèle Faster
    R-CNN avec un backbone ResNet-101.
 Il charge le modèle entraîné, effectue la détection sur une image de test, et retourne les coordonnées des boîtes détectées au format [xmin, ymin, xmax, ymax].
Ces coordonnées peuvent ensuite être utilisées pour recadrer les images de fenêtres et les injecter dans notre modèle de classification ConvNeXT."""

def get_fenet_coordinates(model, image_path, threshold, device):
    """
    Retourne uniquement les coordonnées des boîtes détectées [xmin, ymin, xmax, ymax]
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    
    # On filtre les boîtes selon le seuil de confiance
    keep = scores > threshold
    return boxes[keep] # Retourne directement les boîtes

def main_extraction(model_path, image_test_path):
    device = torch.device("cpu") 
    model = create_model(num_classes=2, pretrained=False)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Appel de la fonction qui renvoie les coordonnées
    coordonnees = get_fenet_coordinates(model, image_test_path, threshold=0.65, device=device)
    
    print(f"Extraction terminée : {len(coordonnees)} fenêtres trouvées.")
    
    # Voici vos coordonnées prêtes à être injectées dans le modèle ConvNeXt
    # Format : [xmin, ymin, xmax, ymax]
    for i, box in enumerate(coordonnees):
        xmin, ymin, xmax, ymax = map(int, box)
        print(f"Fenêtre {i+1} : x={xmin}, y={ymin}, w={xmax-xmin}, h={ymax-ymin}")
        
    return coordonnees

if __name__ == "__main__":
    model_path = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\windows_anatole_files\faster_fenet_resnet101_best.pth"
    image_path = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\essai\SYFW2480.JPG"
    
    coords = main_extraction(model_path, image_path)