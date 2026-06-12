import cv2
import numpy as np
import os

# ==========================================
# PARAMÈTRES ET CHEMINS 
# ==========================================
# Le dossier où se trouvent toutes tes photos originales
#dossier_entree = r"C:\Users\k.nguessan\Desktop\DossierStage\src_anatole_France"
dossier_entree=r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\essai"
# Le dossier où seront sauvegardées les images redressées
#dossier_sortie = r"C:\Users\k.nguessan\Desktop\DossierStage\Dataset_Redresse"
dossier_sortie=r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\essai\Dataset_Redresse"
# Création du dossier de sortie s'il n'existe pas encore
if not os.path.exists(dossier_sortie):
    os.makedirs(dossier_sortie)

nom_fenetre = "Clique sur les 4 coins"

# Variables globales (réinitialisées pour chaque image)
points_cliques = []
image_originale = None
image_affichage = None
image_redressee = None
etape_redressement = False # Permet de savoir si on a fini de cliquer

# ==========================================
# FONCTIONS
# ==========================================
def cliquer_coins(event, x, y, flags, param):
    global points_cliques, image_affichage, etape_redressement
    
    # On n'accepte les clics que si on n'a pas encore atteint 4 points
    if event == cv2.EVENT_LBUTTONDOWN and not etape_redressement:
        if len(points_cliques) < 4:
            points_cliques.append([x, y])
            print(f"Point {len(points_cliques)}/4 : ({x}, {y})")
            
            cv2.circle(image_affichage, (x, y), 7, (0, 0, 255), -1)
            cv2.imshow(nom_fenetre, image_affichage)
            
            # Si on a nos 4 points, on lance le calcul
            if len(points_cliques) == 4:
                etape_redressement = True
                calculer_redressement()

def calculer_redressement():
    global points_cliques, image_originale, image_redressee
    
    pts_source = np.float32(points_cliques)
    (hg, hd, bd, bg) = pts_source
    
    # Calcul dynamique des proportions (sans bord noir)
    largeur_bas = np.sqrt(((bd[0] - bg[0]) ** 2) + ((bd[1] - bg[1]) ** 2))
    largeur_haut = np.sqrt(((hd[0] - hg[0]) ** 2) + ((hd[1] - hg[1]) ** 2))
    largeur = max(int(largeur_bas), int(largeur_haut))
    
    hauteur_droite = np.sqrt(((hd[0] - bd[0]) ** 2) + ((hd[1] - bd[1]) ** 2))
    hauteur_gauche = np.sqrt(((hg[0] - bg[0]) ** 2) + ((hg[1] - bg[1]) ** 2))
    hauteur = max(int(hauteur_droite), int(hauteur_gauche))
    
    pts_destination = np.float32([
        [0, 0], [largeur - 1, 0], [largeur - 1, hauteur - 1], [0, hauteur - 1]
    ])
    
    matrice = cv2.getPerspectiveTransform(pts_source, pts_destination)
    image_redressee = cv2.warpPerspective(image_originale, matrice, (largeur, hauteur))
    
    # Affichage du résultat
    titre_resultat = "2. Facade Redressee"
    cv2.namedWindow(titre_resultat, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(titre_resultat, 800, int(800 * (hauteur/largeur))) # Zoom proportionnel
    cv2.imshow(titre_resultat, image_redressee)
    
    print("\n✅ Redressement terminé !")
    print("👉 's' = Sauvegarder et passer à la suivante")
    print("👉 'n' = Ignorer cette image et passer à la suivante")
    print("👉 'q' = Quitter le programme")

# ==========================================
# 3. BOUCLE PRINCIPALE (TRAITEMENT PAR LOT)
# ==========================================
# On récupère la liste de toutes les images dans le dossier d'entrée
fichiers_images = [f for f in os.listdir(dossier_entree) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not fichiers_images:
    print(f"Erreur : Aucune image trouvée dans {dossier_entree}")
    exit()

print("=====================================================")
print(f"DÉMARRAGE : {len(fichiers_images)} images trouvées à traiter.")
print("=====================================================")

# Initialisation de la fenêtre principale
cv2.namedWindow(nom_fenetre, cv2.WINDOW_NORMAL)
cv2.resizeWindow(nom_fenetre, 1000, 800)
cv2.setMouseCallback(nom_fenetre, cliquer_coins)

# On boucle sur chaque image du dossier
for fichier in fichiers_images:
    chemin_complet = os.path.join(dossier_entree, fichier)
    image_originale = cv2.imread(chemin_complet)
    
    if image_originale is None:
        print(f"Impossible de lire l'image {fichier}. Passage à la suivante.")
        continue

    # Réinitialisation des variables pour la nouvelle image
    image_affichage = image_originale.copy()
    points_cliques = []
    etape_redressement = False
    
    print(f"\n--- Traitement en cours : {fichier} ---")
    cv2.imshow(nom_fenetre, image_affichage)
    
    # Boucle d'attente pour cette image spécifique
    while True:
        # On utilise waitKey(10) pour ne pas bloquer le programme
        # et permettre à la souris de continuer à fonctionner
        touche = cv2.waitKey(10) & 0xFF
        
        # Si le redressement est fait et qu'on appuie sur 's' (Sauvegarder)
        if etape_redressement and touche == ord('s'):
            nom_sortie = f"redressee_{fichier}"
            chemin_sauvegarde = os.path.join(dossier_sortie, nom_sortie)
            cv2.imwrite(chemin_sauvegarde, image_redressee)
            print(f"💾 Sauvegardé sous : {nom_sortie}")
            
            # On ferme la fenêtre de résultat avant de passer à l'image suivante
            cv2.destroyWindow("2. Facade Redressee")
            break # Casse la boucle while pour passer au `fichier` suivant dans le `for`
            
        # Si on veut passer à l'image suivante sans sauvegarder ('n' pour Next/Non)
        elif touche == ord('n'):
            print("⏭️ Image ignorée.")
            if etape_redressement:
                try:
                    cv2.destroyWindow("2. Facade Redressee")
                except:
                    pass
            break # Casse la boucle while pour passer au fichier suivant
            
        # Si on veut quitter complètement l'outil ('q' pour Quitter)
        elif touche == ord('q'):
            print("🛑 Arrêt volontaire du programme.")
            cv2.destroyAllWindows()
            exit() # Arrête complètement le script Python

print("\n🎉 Toutes les images ont été traitées ! Ton dataset est prêt.")
cv2.destroyAllWindows()