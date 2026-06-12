import os
import shutil
import cv2

def trier_base_fenetres(dossier_source, dossier_destination):
    # Définition des classes et des touches associées
    # Touche 'o' -> ouvert, Touche 'f' -> ferme, Touche 'b' -> brouillon
    classes = {
        'o': 'ouvert',
        'f': 'ferme',
        'b': 'brouillon'
    }

    # 1. Création automatique de l'arborescence cible
    for nom_dossier in classes.values():
        os.makedirs(os.path.join(dossier_destination, nom_dossier), exist_ok=True)

    # 2. Récupération de toutes les images du dossier source
    extensions_valides = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    liste_images = [f for f in os.listdir(dossier_source) if f.lower().endswith(extensions_valides)]

    if not liste_images:
        print(f"❌ Aucune image trouvée dans le dossier source : {dossier_source}")
        return

    print("=======================================================")
    print("      OUTIL DE TRI RAPIDE POUR LES FENÊTRES            ")
    print("=======================================================")
    print("Instructions : Regardez l'image affichée et appuyez sur :")
    print("  [o] -> Si la fenêtre est OUVERTE")
    print("  [f] -> Si la fenêtre est FERMÉE")
    print("  [b] -> Si c'est un BROUILLON (mauvais crop, flou, masqué)")
    print("  [q] -> Pour QUITTER et sauvegarder votre progression")
    print("=======================================================\n")
    print(f"Nombre d'images à qualifier : {len(liste_images)}\n")

    for index, nom_img in enumerate(liste_images):
        chemin_complet_source = os.path.join(dossier_source, nom_img)
        
        # Lecture de l'image
        img = cv2.imread(chemin_complet_source)
        if img is None:
            print(f"⚠️ Impossible de lire l'image : {nom_img}, passage à la suivante.")
            continue

        # Redimensionnement uniquement pour le confort visuel de l'affichage (ex: 400x400)
        img_affichage = cv2.resize(img, (400, 400))

        # Affichage à l'écran
        titre_fenetre = f"[{index + 1}/{len(liste_images)}] - {nom_img}"
        cv2.imshow(titre_fenetre, img_affichage)

        # Attente d'une action clavier de l'utilisateur
        while True:
            touche = cv2.waitKey(0) & 0xFF
            touche_char = chr(touche).lower()

            if_quitter = (touche_char == 'q')
            if_action_valide = (touche_char in classes)

            if if_quitter:
                print("\n🛑 Arrêt du programme. Votre progression est sauvegardée.")
                cv2.destroyAllWindows()
                return

            elif if_action_valide:
                dossier_cible = classes[touche_char]
                chemin_complet_cible = os.path.join(dossier_destination, dossier_cible, nom_img)
                
                # Déplacement physique du fichier vers son dossier d'état
                cv2.destroyWindow(titre_fenetre) # Ferme la fenêtre actuelle avant de bouger le fichier
                shutil.move(chemin_complet_source, chemin_complet_cible)
                
                print(f"➡️ Image [{nom_img}] classée dans ──► [{dossier_cible.upper()}]")
                break
            else:
                # Si l'utilisateur appuie sur une autre touche
                print("⚠️ Touche invalide ! Utilisez uniquement [o], [f], [b] ou [q].")

    cv2.destroyAllWindows()
    print("\n🎉 Félicitations ! Toutes les images ont été triées et qualifiées.")

if __name__ == "__main__":
    # --- CONFIGURATION DES CHEMINS ---
    # Mettez ici le dossier où se trouvent actuellement vos images de fenêtres détourées
    DOSSIER_SOURCE_CROP = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\fenetres_detourees"
    
    # Mettez ici le dossier propre où vous voulez voir apparaître les 3 sous-dossiers triés
    DOSSIER_DESTINATION_TRI = r"C:\Users\k.nguessan\Desktop\DocStage\DocStage\Codes\dataset_classification"

    trier_base_fenetres(DOSSIER_SOURCE_CROP, DOSSIER_DESTINATION_TRI)