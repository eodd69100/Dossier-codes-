import cv2 as cv


chemin_image = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\Continuous_PSO.v7i.yolov8\valid\images\redressee_SYFW2345_JPG.rf.99b3d9c0e4161a05f3e8c64d11dee109.jpg"
nom_fenetre = "Outil de Mesure"
# Variables globales pour stocker nos deux points
point1 = None
point2 = None


# FONCTION DE MESURE

def tracer_ligne(event, x, y, flags, param):
    global point1, point2, image
    
    # On écoute UNIQUEMENT le clic gauche (LBUTTONDOWN)
    if event == cv.EVENT_LBUTTONDOWN:
        
        # Si le premier point n'a pas encore été défini
        if point1 is None:
            point1 = (x, y)
            # Dessine un petit cercle rouge plein pour marquer le premier clic
            cv.circle(image, point1, 6, (0, 0, 255), -1)
            cv.imshow(nom_fenetre, image)
            print("Point 1 enregistré. Cliquez sur le 2ème point.")
            
        # Si le point 1 existe déjà, c'est qu'on clique pour le point 2
        elif point2 is None:
            point2 = (x, y)
            # Dessine un petit cercle rouge pour le deuxième clic
            cv.circle(image, point2, 6, (0, 0, 255), -1)
            # Trace une ligne bleue entre les deux points
            cv.line(image, point1, point2, (255, 0, 0), 2)
            cv.imshow(nom_fenetre, image)
            
            # Calcul et affichage de la hauteur instantanément !
            hauteur = abs(point1[1] - point2[1])
            print(f"Ligne tracée ! La hauteur RÉELLE est de : {hauteur} pixels")


# LANCEMENT DU PROGRAMME

# Charger l'image originale
image = cv.imread(chemin_image)

if image is None:
    print(f"Erreur : Impossible de lire l'image au chemin : {chemin_image}")
else:
    
    # Les pixels restent intacts pour une mesure parfaite.

    # Créer la fenêtre avec WINDOW_NORMAL (permet un affichage intelligent)
    cv.namedWindow(nom_fenetre, cv.WINDOW_NORMAL)
    
    # On calcule une belle taille d'affichage (pour éviter le cadre noir)
    hauteur_img, largeur_img = image.shape[:2]
    # On limite l'affichage à 800 pixels de haut maximum, et on ajuste la largeur proportionnellement
    ratio = 800 / hauteur_img
    nouvelle_largeur = int(largeur_img * ratio)
    
    # On redimensionne LA FENÊTRE (et non l'image)
    cv.resizeWindow(nom_fenetre, nouvelle_largeur, 800)

    # Attacher la fonction de souris à notre fenêtre
    cv.setMouseCallback(nom_fenetre, tracer_ligne)

    # Afficher l'image
    print("=====================================================")
    print("MODE D'EMPLOI :")
    print("Cliquez une première fois pour le point haut.")
    print("Cliquez une deuxième fois pour le point bas.")
    print("=====================================================")
    cv.imshow(nom_fenetre, image)

    # Attendre qu'une touche soit pressée pour fermer
    cv.waitKey(0)
    cv.destroyAllWindows()