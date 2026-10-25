# -*- coding: utf-8 -*-  # Indique à Python que ce fichier peut contenir des caractères français.

"""  # Début de la documentation générale du programme.
TP M1 IV - Détection de visage Réel ou Fake avec ACP/PCA.  # Résumé du sujet traité par ce fichier.

Ce programme réalise un système simple d'anti-spoofing facial.  # Il répond à l'objectif du TP.
Il lit automatiquement un dataset placé dans le même dossier que ce fichier Python.  # Le dataset doit être proche du script.
Il transforme chaque image ou frame vidéo en vecteur numérique.  # C'est nécessaire pour appliquer l'ACP.
Il applique l'ACP/PCA pour réduire la dimension.  # L'ACP extrait des caractéristiques plus compactes.
Il entraîne ensuite un classifieur binaire Réel/Fake.  # Le modèle apprend à distinguer les deux classes.
Il affiche les métriques de performance et sauvegarde le modèle.  # Les résultats peuvent être utilisés dans le rapport.

Convention utilisée dans ce code :  # Cette partie clarifie les labels.
    label 1 = Réel / Live / Genuine / Bonafide  # Classe positive.
    label 0 = Fake / Spoof / Attack / Photo / Replay / Mask  # Classe négative.

Important :  # Avertissement utile pour CASIA.
    Si votre dataset CASIA est organisé avec des noms différents,  # Certaines versions changent les dossiers.
    modifiez uniquement les listes MOTS_CLES_REEL et MOTS_CLES_FAKE,  # C'est le plus simple.
    ou bien placez les images dans deux dossiers nommés real et fake.  # Organisation recommandée.
"""  # Fin de la documentation générale du programme.

from pathlib import Path  # Sert à manipuler les chemins de fichiers de manière propre et portable.
import argparse  # Sert à ajouter des options en ligne de commande comme --image.
import re  # Sert à rechercher des nombres ou des mots dans les noms de fichiers.
import sys  # Sert à arrêter proprement le programme en cas d'erreur.
import warnings  # Sert à afficher des avertissements sans arrêter le programme.

import cv2  # OpenCV sert à lire les images, les vidéos et détecter les visages.
import joblib  # Sert à sauvegarder et recharger le modèle entraîné.
import matplotlib.pyplot as plt  # Sert à produire les figures des résultats.
import numpy as np  # Sert aux calculs numériques et aux tableaux de données.
import pandas as pd  # Sert surtout à lire un fichier CSV de labels si le dataset en possède un.

from matplotlib.patches import Ellipse  # Sert à dessiner les ellipses de confiance sur le graphe ACP.

from sklearn.decomposition import PCA  # Implémente l'ACP, appelée PCA en anglais.
from sklearn.metrics import accuracy_score  # Calcule le taux de bonnes prédictions.
from sklearn.metrics import classification_report  # Donne précision, rappel et F1-score.
from sklearn.metrics import confusion_matrix  # Calcule la matrice de confusion.
from sklearn.metrics import ConfusionMatrixDisplay  # Affiche proprement la matrice de confusion.
from sklearn.model_selection import train_test_split  # Sépare le dataset en apprentissage et test.
from sklearn.naive_bayes import GaussianNB  # Classifieur bayésien gaussien pour comparaison avec SVM.
from sklearn.pipeline import Pipeline  # Chaîne StandardScaler + PCA + SVM dans un seul modèle.
from sklearn.preprocessing import StandardScaler  # Standardise les variables avant l'ACP.
from sklearn.svm import SVC  # Classifieur SVM utilisé après l'ACP.

# 1) PARAMÈTRES GÉNÉRAUX DU TP   # Partie configurable du programme.

BASE_DIR = Path(__file__).resolve().parent  # Dossier où se trouve ce fichier Python.
DATASET_DIR = BASE_DIR  # On suppose que le dataset est dans le même répertoire que le programme.
DOSSIER_SORTIE = BASE_DIR / "resultats_tp_acp"  # Dossier dans lequel on sauvegarde les résultats.
MODELE_SORTIE = DOSSIER_SORTIE / "modele_acp_anti_spoofing.joblib"  # Fichier du modèle sauvegardé.

TAILLE_IMAGE = (96, 96)  # Taille finale de chaque visage : 96x96 pixels pour limiter la dimension.
UTILISER_DETECTION_VISAGE = True  # True = on essaie de couper le visage avant l'ACP.
N_COMPONENTS_ACP = 0.95  # L'ACP garde assez d'axes pour conserver 95% de variance.
TAILLE_TEST = 0.25  # 25% des données sont gardées pour tester le modèle.
RANDOM_STATE = 42  # Graine aléatoire pour rendre les résultats reproductibles.
FRAME_STEP_VIDEO = 15  # Pour une vidéo, on prend une frame chaque 15 frames afin d'éviter trop de redondance.
MAX_FRAMES_PAR_VIDEO = 40  # Nombre maximum de frames extraites par vidéo.
MAX_ECHANTILLONS_TOTAL = None  # Mettre un entier comme 2000 pour limiter rapidement un très grand dataset.

EXTENSIONS_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}  # Formats image acceptés.
EXTENSIONS_VIDEOS = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}  # Formats vidéo acceptés.

MOTS_CLES_REEL = [  # Mots qui indiquent généralement un visage réel.
    "real",  # Mot anglais courant.
    "reel",  # Variante sans accent.
    "réel",  # Variante avec accent.
    "live",  # Souvent utilisé dans les datasets anti-spoofing.
    "genuine",  # Terme anglais signifiant authentique.
    "bonafide",  # Terme utilisé dans plusieurs datasets biométriques.
    "vrai",  # Mot français.
    "normal",  # Souvent utilisé pour les vidéos réelles.
]

MOTS_CLES_FAKE = [  # Mots qui indiquent généralement une attaque ou un visage falsifié.
    "fake",  # Mot anglais courant.
    "faux",  # Mot français.
    "spoof",  # Terme classique en anti-spoofing.
    "attack",  # Terme anglais pour attaque.
    "attaque",  # Variante française.
    "photo",  # Attaque par photo imprimée.
    "print",  # Attaque par impression papier.
    "printed",  # Variante de print.
    "replay",  # Attaque par écran ou vidéo rejouée.
    "screen",  # Attaque via écran.
    "mask",  # Attaque par masque.
    "warped",  # Photo courbée ou déformée dans CASIA.
    "cut",  # Photo découpée dans CASIA.
]

LABELS_TEXTE = {0: "Fake", 1: "Reel"}  

# 2) FONCTIONS D'AFFICHAGE       # Fonctions utiles pour rendre la sortie plus claire.


def afficher_titre(titre: str) -> None:  # Déclare une fonction qui affiche un titre dans la console.
    """Affiche un titre clair pour séparer les étapes du programme."""  # Explique le rôle de la fonction.
    print("\n" + "=" * 80)  # Affiche une ligne de séparation avant le titre.
    print(titre)  # Affiche le texte du titre.
    print("=" * 80)  # Affiche une ligne de séparation après le titre.


def afficher_erreur_dataset() -> None:  # Déclare une fonction appelée si aucun dataset exploitable n'est trouvé.
    """Explique à l'utilisateur comment organiser son dataset."""  # Explique le rôle de la fonction.
    print("\nAucune donnée exploitable n'a été trouvée.")  # Informe que le chargement a échoué.
    print("Placez le dataset dans le même dossier que ce fichier Python.")  # Rappelle la contrainte du sujet.
    print("Organisation recommandée :")  # Introduit l'exemple d'organisation.
    print("  dossier_du_programme/")  # Montre le dossier principal.
    print("  ├── tp_acp_anti_spoofing.py")  # Montre le fichier Python.
    print("  ├── real/        -> images ou vidéos de vrais visages")  # Montre le dossier des réels.
    print("  └── fake/        -> images ou vidéos de visages falsifiés")  # Montre le dossier des fakes.
    print("\nAutre possibilité :")  # Introduit une autre organisation possible.
    print("  Utiliser des sous-dossiers contenant des mots comme real, live, fake, spoof, attack, photo.")  # Explique les mots-clés.
    print("\nSi votre dataset CASIA contient seulement des noms comme 1.avi, 2.avi, etc.,")  # Mentionne le cas CASIA.
    print("vérifiez la convention de votre version et adaptez la fonction inferer_label_depuis_chemin().")  # Donne la fonction à modifier.


# 3) FONCTIONS DE LABELISATION   # Fonctions qui trouvent si un fichier est réel ou fake.

def nettoyer_texte(texte: str) -> str:  # Déclare une fonction qui normalise un texte.
    """Transforme un texte en minuscule simple pour faciliter la recherche de mots-clés."""  # Explique le rôle.
    texte = texte.lower()  # Convertit le texte en minuscules.
    texte = texte.replace("é", "e")  # Remplace é par e pour éviter les problèmes d'accents.
    texte = texte.replace("è", "e")  # Remplace è par e pour éviter les problèmes d'accents.
    texte = texte.replace("ê", "e")  # Remplace ê par e pour éviter les problèmes d'accents.
    texte = texte.replace("à", "a")  # Remplace à par a pour éviter les problèmes d'accents.
    texte = texte.replace("_", " ")  # Remplace les underscores par des espaces.
    texte = texte.replace("-", " ")  # Remplace les tirets par des espaces.
    return texte  # Retourne le texte nettoyé.


def convertir_label_csv(label_brut) -> int | None:  # Déclare une fonction qui convertit un label venant d'un CSV.
    """Convertit plusieurs écritures possibles du label en 0 ou 1."""  # Explique le rôle.
    if pd.isna(label_brut):  # Vérifie si la cellule du CSV est vide.
        return None  # Retourne None car le label est inconnu.
    texte = nettoyer_texte(str(label_brut))  # Convertit le label en texte nettoyé.
    if texte.strip() in {"1", "real", "reel", "live", "genuine", "bonafide", "vrai", "normal"}:  # Teste les labels réels.
        return 1  # Retourne 1 pour un visage réel.
    if texte.strip() in {"0", "fake", "faux", "spoof", "attack", "attaque", "photo", "replay", "mask"}:  # Teste les labels fake.
        return 0  # Retourne 0 pour un visage fake.
    return None  # Retourne None si le label n'est pas reconnu.


def inferer_label_depuis_chemin(chemin: Path) -> int | None:  # Déclare une fonction qui déduit le label à partir du chemin.
    """Déduit automatiquement le label d'une image ou vidéo grâce aux noms de dossiers et fichiers."""  # Explique le rôle.
    chemin_texte = nettoyer_texte(str(chemin))  # Transforme tout le chemin en texte nettoyé.
    for mot in MOTS_CLES_FAKE:  # Parcourt les mots-clés de la classe fake.
        if mot in chemin_texte:  # Vérifie si le mot-clé apparaît dans le chemin.
            return 0  # Retourne 0 si le chemin correspond à une attaque.
    for mot in MOTS_CLES_REEL:  # Parcourt les mots-clés de la classe réelle.
        if mot in chemin_texte:  # Vérifie si le mot-clé apparaît dans le chemin.
            return 1  # Retourne 1 si le chemin correspond à un visage réel.

    numero = extraire_numero_casia(chemin)  # Essaie d'extraire un numéro de vidéo CASIA.
    if numero is not None:  # Vérifie si un numéro a été trouvé.
        if numero in {1, 2, 3}:  # Convention fréquente : les trois premières vidéos sont réelles.
            warnings.warn(f"Label CASIA inféré comme Réel pour {chemin.name}; vérifiez la convention de votre dataset.")  # Avertit l'utilisateur.
            return 1  # Retourne réel selon cette convention fréquente.
        if 4 <= numero <= 12:  # Convention fréquente : les vidéos 4 à 12 sont des attaques.
            warnings.warn(f"Label CASIA inféré comme Fake pour {chemin.name}; vérifiez la convention de votre dataset.")  # Avertit l'utilisateur.
            return 0  # Retourne fake selon cette convention fréquente.

    return None  # Retourne None si aucun label fiable n'a été trouvé.


def extraire_numero_casia(chemin: Path) -> int | None:  # Déclare une fonction qui récupère un numéro dans le nom de fichier.
    """Essaie d'extraire un numéro de fichier CASIA comme 1.avi, 2.avi, HR_1.avi, etc."""  # Explique le rôle.
    nom = chemin.stem.lower()  # Récupère le nom du fichier sans extension.
    nombres = re.findall(r"\d+", nom)  # Cherche tous les nombres présents dans le nom.
    if not nombres:  # Vérifie si aucun nombre n'a été trouvé.
        return None  # Retourne None car il n'y a pas de numéro.
    return int(nombres[-1])  # Retourne le dernier nombre trouvé dans le nom

# 4) CHARGEMENT DU DATASET       # Fonctions qui lisent les images ou vidéos

def trouver_fichier_csv_labels(dossier: Path) -> Path | None:  # Déclare une fonction qui cherche un CSV de labels.
    """Cherche un fichier CSV pouvant contenir les chemins d'images et leurs labels."""  # Explique le rôle.
    candidats = list(dossier.rglob("*.csv"))  # Cherche tous les fichiers CSV dans le dossier du dataset.
    for csv_path in candidats:  # Parcourt chaque CSV trouvé.
        if csv_path.parent == DOSSIER_SORTIE:  # Ignore le dossier de résultats du programme.
            continue  # Passe au CSV suivant.
        try:  # Essaie de lire le CSV.
            df = pd.read_csv(csv_path, nrows=5)  # Lit uniquement les premières lignes pour inspecter les colonnes.
        except Exception:  # Capture les erreurs de lecture CSV.
            continue  # Ignore ce CSV s'il n'est pas lisible.
        colonnes = {c.lower().strip() for c in df.columns}  # Convertit les noms de colonnes en minuscules.
        colonnes_chemin = {"image", "img", "path", "filepath", "file", "filename", "video"}  # Colonnes possibles pour les chemins.
        colonnes_label = {"label", "class", "target", "type", "y"}  # Colonnes possibles pour les labels.
        if colonnes.intersection(colonnes_chemin) and colonnes.intersection(colonnes_label):  # Vérifie si le CSV semble utile.
            return csv_path  # Retourne le premier CSV compatible.
    return None  # Retourne None si aucun CSV de labels n'est trouvé.


def charger_references_depuis_csv(csv_path: Path) -> list[tuple[Path, int]]:  # Déclare une fonction qui lit les références depuis un CSV.
    """Charge une liste de fichiers et labels à partir d'un fichier CSV."""  # Explique le rôle.
    df = pd.read_csv(csv_path)  # Lit tout le CSV de labels.
    colonnes_originales = list(df.columns)  # Garde les noms originaux des colonnes.
    colonnes_min = {c.lower().strip(): c for c in colonnes_originales}  # Associe nom minuscule -> nom réel.
    colonne_chemin = None  # Initialise la colonne contenant les chemins.
    colonne_label = None  # Initialise la colonne contenant les labels.

    for nom in ["image", "img", "path", "filepath", "file", "filename", "video"]:  # Teste les noms possibles de chemin.
        if nom in colonnes_min:  # Vérifie si ce nom existe dans le CSV.
            colonne_chemin = colonnes_min[nom]  # Garde le vrai nom de la colonne.
            break  # Arrête la recherche car on a trouvé.

    for nom in ["label", "class", "target", "type", "y"]:  # Teste les noms possibles de label.
        if nom in colonnes_min:  # Vérifie si ce nom existe dans le CSV.
            colonne_label = colonnes_min[nom]  # Garde le vrai nom de la colonne.
            break  # Arrête la recherche car on a trouvé.

    if colonne_chemin is None or colonne_label is None:  # Vérifie si une colonne essentielle manque.
        return []  # Retourne une liste vide si le CSV n'est pas exploitable.

    references = []  # Initialise la liste des couples fichier-label.
    for _, ligne in df.iterrows():  # Parcourt chaque ligne du CSV.
        label = convertir_label_csv(ligne[colonne_label])  # Convertit le label en 0 ou 1.
        if label is None:  # Vérifie si le label n'a pas été reconnu.
            continue  # Ignore cette ligne.
        chemin_relatif = Path(str(ligne[colonne_chemin]))  # Convertit le chemin lu en objet Path.
        chemin_fichier = chemin_relatif if chemin_relatif.is_absolute() else csv_path.parent / chemin_relatif  # Rend le chemin absolu si besoin.
        if chemin_fichier.exists():  # Vérifie que le fichier existe réellement.
            references.append((chemin_fichier, label))  # Ajoute le couple fichier-label.
    return references  # Retourne toutes les références trouvées.


def lister_fichiers_dataset(dossier: Path) -> list[tuple[Path, int]]:  # Déclare une fonction qui liste les fichiers et labels.
    """Liste les images/vidéos du dataset et leur associe automatiquement un label."""  # Explique le rôle.
    csv_labels = trouver_fichier_csv_labels(dossier)  # Cherche d'abord un CSV de labels.
    if csv_labels is not None:  # Vérifie si un CSV compatible a été trouvé.
        references_csv = charger_references_depuis_csv(csv_labels)  # Charge les chemins et labels du CSV.
        if references_csv:  # Vérifie si le CSV a donné des références valides.
            print(f"CSV de labels utilisé : {csv_labels}")  # Informe l'utilisateur que le CSV est utilisé.
            return references_csv  # Retourne directement les références du CSV.

    references = []  # Initialise la liste des fichiers labellisés automatiquement.
    extensions = EXTENSIONS_IMAGES.union(EXTENSIONS_VIDEOS)  # Regroupe les extensions images et vidéos.
    for chemin in dossier.rglob("*"):  # Parcourt récursivement tous les fichiers du dossier.
        if not chemin.is_file():  # Vérifie que l'élément est bien un fichier.
            continue  # Ignore les dossiers.
        if DOSSIER_SORTIE in chemin.parents:  # Évite de relire les résultats du programme.
            continue  # Ignore les fichiers générés par le programme.
        if chemin.suffix.lower() not in extensions:  # Vérifie si le format est accepté.
            continue  # Ignore les fichiers qui ne sont ni images ni vidéos.
        label = inferer_label_depuis_chemin(chemin)  # Essaie de trouver le label à partir du chemin.
        if label is None:  # Vérifie si le label est inconnu.
            continue  # Ignore les fichiers sans label clair.
        references.append((chemin, label))  # Ajoute le fichier et son label.
    return references  # Retourne la liste complète des fichiers labellisés.


def creer_detecteur_visage():  # Déclare une fonction qui prépare le détecteur de visage OpenCV.
    """Crée un détecteur Haar Cascade pour localiser le visage dans une image."""  # Explique le rôle.
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"  # Chemin du modèle Haar fourni par OpenCV.
    if not cascade_path.exists():  # Vérifie si le fichier du détecteur existe.
        return None  # Retourne None si OpenCV ne fournit pas le fichier.
    detecteur = cv2.CascadeClassifier(str(cascade_path))  # Charge le détecteur de visage.
    if detecteur.empty():  # Vérifie si le détecteur a échoué à charger.
        return None  # Retourne None si le détecteur n'est pas utilisable.
    return detecteur  # Retourne le détecteur prêt à l'emploi.


def extraire_visage_ou_image(image_bgr: np.ndarray, detecteur) -> np.ndarray:  # Déclare une fonction de découpage du visage.
    """Retourne la zone du visage si elle est détectée, sinon retourne l'image entière."""  # Explique le rôle.
    if not UTILISER_DETECTION_VISAGE:  # Vérifie si l'utilisateur veut désactiver la détection.
        return image_bgr  # Retourne l'image complète.
    if detecteur is None:  # Vérifie si aucun détecteur n'est disponible.
        return image_bgr  # Retourne l'image complète.

    gris = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)  # Convertit l'image en gris pour la détection Haar.
    visages = detecteur.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))  # Détecte les visages.
    if len(visages) == 0:  # Vérifie si aucun visage n'a été détecté.
        return image_bgr  # Retourne l'image entière dans ce cas.

    x, y, w, h = max(visages, key=lambda boite: boite[2] * boite[3])  # Choisit le plus grand visage détecté.
    marge_x = int(0.10 * w)  # Calcule une marge horizontale de 10%.
    marge_y = int(0.10 * h)  # Calcule une marge verticale de 10%.
    x1 = max(0, x - marge_x)  # Calcule la limite gauche en restant dans l'image.
    y1 = max(0, y - marge_y)  # Calcule la limite haute en restant dans l'image.
    x2 = min(image_bgr.shape[1], x + w + marge_x)  # Calcule la limite droite en restant dans l'image.
    y2 = min(image_bgr.shape[0], y + h + marge_y)  # Calcule la limite basse en restant dans l'image.
    return image_bgr[y1:y2, x1:x2]  # Retourne la zone du visage avec une petite marge.


def preparer_image(image_bgr: np.ndarray, detecteur) -> np.ndarray:  # Déclare une fonction qui transforme une image en vecteur.
    """Prétraite une image : visage, gris, redimensionnement, normalisation et vectorisation."""  # Explique le rôle.
    region = extraire_visage_ou_image(image_bgr, detecteur)  # Découpe le visage si possible.
    gris = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)  # Convertit l'image en niveaux de gris.
    gris = cv2.resize(gris, TAILLE_IMAGE, interpolation=cv2.INTER_AREA)  # Redimensionne l'image à la taille fixe.
    gris = gris.astype(np.float32) / 255.0  # Normalise les pixels entre 0 et 1.
    vecteur = gris.flatten()  # Transforme la matrice 2D en vecteur 1D pour l'ACP.
    return vecteur  # Retourne le vecteur numérique représentant l'image.


def charger_image_fichier(chemin: Path, detecteur) -> np.ndarray | None:  # Déclare une fonction qui lit une image disque.
    """Lit une image depuis le disque et retourne son vecteur prétraité."""  # Explique le rôle.
    image_bgr = cv2.imread(str(chemin))  # Lit l'image avec OpenCV au format BGR.
    if image_bgr is None:  # Vérifie si l'image n'a pas pu être lue.
        return None  # Retourne None si lecture impossible.
    return preparer_image(image_bgr, detecteur)  # Prétraite l'image et retourne son vecteur.


def extraire_vecteurs_video(chemin: Path, detecteur) -> list[np.ndarray]:  # Déclare une fonction qui extrait des frames d'une vidéo.
    """Extrait plusieurs frames d'une vidéo et retourne leurs vecteurs prétraités."""  # Explique le rôle.
    capture = cv2.VideoCapture(str(chemin))  # Ouvre la vidéo avec OpenCV.
    if not capture.isOpened():  # Vérifie si la vidéo n'a pas pu être ouverte.
        return []  # Retourne une liste vide si lecture impossible.

    vecteurs = []  # Initialise la liste des vecteurs extraits.
    indice_frame = 0  # Initialise le compteur de frames.
    while True:  # Boucle de lecture frame par frame.
        succes, frame = capture.read()  # Lit une frame de la vidéo.
        if not succes:  # Vérifie si la vidéo est terminée ou illisible.
            break  # Sort de la boucle.
        if indice_frame % FRAME_STEP_VIDEO == 0:  # Garde seulement une frame toutes les FRAME_STEP_VIDEO frames.
            vecteur = preparer_image(frame, detecteur)  # Prétraite la frame comme une image.
            vecteurs.append(vecteur)  # Ajoute le vecteur à la liste.
        if len(vecteurs) >= MAX_FRAMES_PAR_VIDEO:  # Vérifie si on a déjà extrait assez de frames.
            break  # Arrête l'extraction pour cette vidéo.
        indice_frame += 1  # Passe à la frame suivante.

    capture.release()  # Libère correctement la vidéo.
    return vecteurs  # Retourne les vecteurs extraits.


def construire_dataset(dossier: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:  # Déclare une fonction qui construit X et y.
    """Construit la matrice X des images vectorisées et le vecteur y des labels."""  # Explique le rôle.
    afficher_titre("Étape 1 - Chargement du dataset")  # Affiche le titre de l'étape.
    detecteur = creer_detecteur_visage()  # Crée le détecteur de visage.
    if detecteur is None and UTILISER_DETECTION_VISAGE:  # Vérifie si la détection demandée est indisponible.
        print("Détecteur de visage indisponible : le programme utilisera l'image entière.")  # Informe l'utilisateur.

    references = lister_fichiers_dataset(dossier)  # Liste les fichiers et leurs labels.
    if not references:  # Vérifie si aucun fichier n'a été trouvé.
        afficher_erreur_dataset()  # Affiche les instructions d'organisation.
        sys.exit(1)  # Arrête le programme avec un code d'erreur.

    X = []  # Liste qui contiendra les vecteurs images.
    y = []  # Liste qui contiendra les labels.
    sources = []  # Liste qui contiendra les chemins sources pour traçabilité.

    for chemin, label in references:  # Parcourt chaque image ou vidéo labellisée.
        if MAX_ECHANTILLONS_TOTAL is not None and len(X) >= MAX_ECHANTILLONS_TOTAL:  # Vérifie la limite globale éventuelle.
            break  # Arrête le chargement si la limite est atteinte.
        extension = chemin.suffix.lower()  # Récupère l'extension du fichier.
        if extension in EXTENSIONS_IMAGES:  # Vérifie si le fichier est une image.
            vecteur = charger_image_fichier(chemin, detecteur)  # Charge et vectorise l'image.
            if vecteur is None:  # Vérifie si l'image n'a pas pu être lue.
                continue  # Ignore l'image illisible.
            X.append(vecteur)  # Ajoute le vecteur image.
            y.append(label)  # Ajoute le label correspondant.
            sources.append(str(chemin))  # Garde le chemin de l'image.
        elif extension in EXTENSIONS_VIDEOS:  # Vérifie si le fichier est une vidéo.
            vecteurs = extraire_vecteurs_video(chemin, detecteur)  # Extrait plusieurs frames vectorisées.
            for indice, vecteur in enumerate(vecteurs):  # Parcourt les frames extraites.
                if MAX_ECHANTILLONS_TOTAL is not None and len(X) >= MAX_ECHANTILLONS_TOTAL:  # Vérifie la limite globale.
                    break  # Arrête l'ajout si la limite est atteinte.
                X.append(vecteur)  # Ajoute le vecteur de la frame.
                y.append(label)  # Ajoute le label hérité de la vidéo.
                sources.append(f"{chemin}::frame_{indice}")  # Garde la source précise de la frame.

    X = np.array(X, dtype=np.float32)  # Convertit la liste des vecteurs en matrice NumPy.
    y = np.array(y, dtype=np.int64)  # Convertit les labels en tableau NumPy.

    print(f"Nombre total d'échantillons chargés : {len(X)}")  # Affiche le nombre d'échantillons.
    print(f"Dimension initiale d'un échantillon : {X.shape[1]} pixels")  # Affiche la dimension avant ACP.
    print(f"Nombre d'échantillons Réel : {np.sum(y == 1)}")  # Affiche le nombre de réels.
    print(f"Nombre d'échantillons Fake : {np.sum(y == 0)}")  # Affiche le nombre de fakes.

    verifier_dataset(X, y)  # Vérifie que le dataset est exploitable.
    return X, y, sources  # Retourne la matrice X, les labels y et les chemins sources.


def verifier_dataset(X: np.ndarray, y: np.ndarray) -> None:  # Déclare une fonction de validation du dataset.
    """Vérifie que le dataset contient assez de données et les deux classes."""  # Explique le rôle.
    if len(X) == 0:  # Vérifie si aucune donnée n'a été chargée.
        raise ValueError("Aucune image ou frame n'a été chargée.")  # Signale une erreur claire.
    classes = np.unique(y)  # Récupère les classes présentes.
    if len(classes) < 2:  # Vérifie si une seule classe est présente.
        raise ValueError("Le dataset doit contenir au moins deux classes : Réel et Fake.")  # Signale une erreur claire.
    if len(X) < 10:  # Vérifie si le nombre d'échantillons est très faible.
        warnings.warn("Dataset très petit : les résultats risquent de ne pas être fiables.")  # Avertit sans arrêter.


# =============================  # Séparateur visuel.
# 5) ACP + CLASSIFICATION        # Fonctions qui entraînent et évaluent le modèle.
# =============================  # Séparateur visuel.

def creer_pipeline_acp_svm() -> Pipeline:  # Déclare une fonction qui crée le modèle complet.
    """Crée la chaîne de traitement : standardisation -> ACP -> SVM."""  # Explique le rôle.
    pipeline = Pipeline(  # Crée un pipeline scikit-learn.
        steps=[  # Liste ordonnée des étapes du pipeline.
            ("standardisation", StandardScaler()),  # Met les variables à une échelle comparable.
            ("acp", PCA(n_components=N_COMPONENTS_ACP, svd_solver="full", random_state=RANDOM_STATE)),  # Réduit la dimension en gardant 95% de variance.
            ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)),  # Classe Réel/Fake avec un SVM robuste.
        ]  # Fin de la liste des étapes.
    )  # Fin de la création du pipeline.
    return pipeline  # Retourne le modèle complet.


# =============================  # Séparateur visuel.
# 5b) ANALYSE ACP APPROFONDIE   # Visualisations dédiées avant la classification.
# =============================  # Séparateur visuel.
# Note pédagogique : L'ACP ne sert pas uniquement à réduire la dimension avant le SVM.
# Elle permet aussi d'explorer visuellement les données, de comprendre leur structure,
# d'identifier des outliers et de valider l'hypothèse de séparabilité des classes.
# Ces graphes doivent être produits AVANT la classification et commentés dans le rapport.


def plot_ellipse(ax, mean: np.ndarray, cov: np.ndarray, n_std: float = 2.0, **kwargs) -> None:
    """Dessine une ellipse de confiance à n_std écarts-types sur un axe Matplotlib.

    L'ellipse représente la région de confiance à 95 % (n_std=2) pour une loi normale
    bivariée de moyenne `mean` et de matrice de covariance `cov`.

    Paramètres
    ----------
    ax : matplotlib.axes.Axes
        Axe sur lequel l'ellipse est dessinée.
    mean : np.ndarray de forme (2,)
        Centre de l'ellipse (moyenne PC1, PC2 de la classe).
    cov : np.ndarray de forme (2, 2)
        Matrice de covariance 2D de la classe.
    n_std : float
        Nombre d'écarts-types de rayon (2.0 ≈ 95 % sous hypothèse normale).
    **kwargs
        Paramètres visuels transmis à matplotlib.patches.Ellipse (couleur, alpha, etc.).
    """
    valeurs_propres, vecteurs_propres = np.linalg.eigh(cov)  # Calcule les valeurs et vecteurs propres de cov.
    ordre = valeurs_propres.argsort()[::-1]  # Trie les valeurs propres par ordre décroissant.
    valeurs_propres = valeurs_propres[ordre]  # Réordonne les valeurs propres.
    vecteurs_propres = vecteurs_propres[:, ordre]  # Réordonne les vecteurs propres.

    angle = np.degrees(np.arctan2(vecteurs_propres[1, 0], vecteurs_propres[0, 0]))  # Calcule l'angle de l'ellipse en degrés.
    largeur = 2.0 * n_std * np.sqrt(np.abs(valeurs_propres[0]))  # Largeur de l'ellipse (axe principal).
    hauteur = 2.0 * n_std * np.sqrt(np.abs(valeurs_propres[1]))  # Hauteur de l'ellipse (axe secondaire).

    ellipse = Ellipse(  # Crée l'objet ellipse Matplotlib.
        xy=mean,  # Centre de l'ellipse.
        width=largeur,  # Largeur de l'ellipse.
        height=hauteur,  # Hauteur de l'ellipse.
        angle=angle,  # Orientation de l'ellipse.
        fill=False,  # Ellipse vide (contour uniquement).
        linewidth=2,  # Épaisseur du contour.
        **kwargs,  # Autres paramètres visuels.
    )
    ax.add_patch(ellipse)  # Ajoute l'ellipse à l'axe fourni.


def analyse_acp_approfondie(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dossier_sortie: Path,
) -> dict:
    """Réalise une analyse ACP exploratoire sur les données d'apprentissage.

    Cette fonction est appelée AVANT la classification SVM pour explorer la structure
    des données dans l'espace des composantes principales. Elle produit trois figures :
      1. Graphe des individus (PC1 vs PC2) avec ellipses de confiance à 95 %.
      2. Cercle des corrélations sur les variables les plus contributives à PC1.
      3. Barplot des 20 variables les plus contributives à PC1.

    Elle calcule également la distance de Mahalanobis entre les deux centres de classe,
    et compare le SVM avec un classifieur bayésien gaussien (GaussianNB) sur l'espace ACP.

    Paramètres
    ----------
    X_train : np.ndarray
        Matrice des échantillons d'apprentissage (avant standardisation).
    y_train : np.ndarray
        Labels des échantillons d'apprentissage (0 = Fake, 1 = Reel).
    dossier_sortie : Path
        Dossier dans lequel les figures PNG seront sauvegardées.

    Retourne
    -------
    dict
        Dictionnaire contenant les clés :
          - 'distance_mahalanobis' : float, distance entre les deux centres de classe.
          - 'variance_pc1' : float, proportion de variance expliquée par PC1.
          - 'variance_pc2' : float, proportion de variance expliquée par PC2.
          - 'n_composantes' : int, nombre total de composantes retenues.
          - 'accuracy_gnb' : float, accuracy du classifieur bayésien gaussien.
    """
    dossier_sortie = Path(dossier_sortie)  # S'assure que le dossier est bien un objet Path.
    dossier_sortie.mkdir(exist_ok=True)  # Crée le dossier de sortie si nécessaire.

    # --- Standardisation et ACP ---
    scaler = StandardScaler()  # Crée le scaler pour mettre les variables à l'échelle.
    X_std = scaler.fit_transform(X_train)  # Standardise les données d'apprentissage.

    acp = PCA(n_components=N_COMPONENTS_ACP, svd_solver="full", random_state=RANDOM_STATE)  # Crée l'ACP avec les mêmes paramètres que le pipeline.
    X_pca = acp.fit_transform(X_std)  # Projette les données standardisées sur les axes principaux.

    variance_pc1 = float(acp.explained_variance_ratio_[0])  # Variance expliquée par la première composante.
    variance_pc2 = float(acp.explained_variance_ratio_[1]) if acp.n_components_ > 1 else 0.0  # Variance de la deuxième composante.
    n_composantes = int(acp.n_components_)  # Nombre total de composantes retenues.

    # --- Séparation par classe pour les deux premières composantes ---
    X_fake = X_pca[y_train == 0, :2]  # Points de la classe Fake dans le plan PC1-PC2.
    X_reel = X_pca[y_train == 1, :2]  # Points de la classe Reel dans le plan PC1-PC2.

    # --- Statistiques par classe pour les ellipses ---
    distance_mahalanobis = 0.0  # Valeur par défaut si une classe est trop petite.
    mean_fake = mean_reel = None  # Initialisations des centres.
    cov_fake = cov_reel = None  # Initialisations des matrices de covariance.

    if len(X_fake) >= 2:  # Vérifie qu'on a assez d'échantillons Fake pour calculer la covariance.
        mean_fake = np.mean(X_fake, axis=0)  # Centre de la classe Fake dans le plan ACP.
        cov_fake = np.cov(X_fake.T)  # Matrice de covariance 2D de la classe Fake.

    if len(X_reel) >= 2:  # Vérifie qu'on a assez d'échantillons Reel pour calculer la covariance.
        mean_reel = np.mean(X_reel, axis=0)  # Centre de la classe Reel dans le plan ACP.
        cov_reel = np.cov(X_reel.T)  # Matrice de covariance 2D de la classe Reel.

    # --- Distance de Mahalanobis entre les deux centres de classe ---
    if mean_fake is not None and mean_reel is not None and cov_fake is not None and cov_reel is not None:
        n0 = len(X_fake)  # Nombre d'échantillons Fake.
        n1 = len(X_reel)  # Nombre d'échantillons Reel.
        cov_pooled = ((n0 - 1) * cov_fake + (n1 - 1) * cov_reel) / (n0 + n1 - 2)  # Covariance poolée (combinée).
        diff = mean_reel - mean_fake  # Différence entre les deux centres de classe.
        try:
            distance_mahalanobis = float(np.sqrt(diff @ np.linalg.inv(cov_pooled) @ diff))  # Calcule la distance de Mahalanobis.
        except np.linalg.LinAlgError:  # En cas de matrice singulière.
            distance_mahalanobis = 0.0  # Retourne 0 si le calcul est impossible.

    # =====================================================================
    # FIGURE 1 : Graphe des individus avec ellipses de confiance à 95 %
    # =====================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 7))  # Crée la figure du graphe des individus.

    couleurs = {0: "#e74c3c", 1: "#2980b9"}  # Rouge pour Fake, bleu pour Reel.
    marqueurs = {0: "o", 1: "^"}  # Cercle pour Fake, triangle pour Reel.

    for label, nom, couleur, marqueur in [
        (0, "Fake", couleurs[0], marqueurs[0]),
        (1, "Réel", couleurs[1], marqueurs[1]),
    ]:
        masque = y_train == label  # Sélectionne les échantillons de cette classe.
        ax1.scatter(  # Trace les points de la classe.
            X_pca[masque, 0],  # Coordonnée sur PC1.
            X_pca[masque, 1],  # Coordonnée sur PC2.
            c=couleur,  # Couleur de la classe.
            marker=marqueur,  # Marqueur de la classe.
            alpha=0.45,  # Transparence pour voir les superpositions.
            s=18,  # Taille des points.
            label=f"{nom} (n={np.sum(masque)})",  # Légende avec effectif.
        )

    # Dessin des ellipses de confiance à 95 %
    if mean_fake is not None and cov_fake is not None and len(X_fake) >= 2:
        plot_ellipse(ax1, mean_fake, cov_fake, n_std=2, color=couleurs[0], linestyle="--", label="Ellipse 95% Fake")  # Ellipse Fake.
    if mean_reel is not None and cov_reel is not None and len(X_reel) >= 2:
        plot_ellipse(ax1, mean_reel, cov_reel, n_std=2, color=couleurs[1], linestyle="--", label="Ellipse 95% Réel")  # Ellipse Reel.

    titre_dist = f"  |  Distance de Mahalanobis : {distance_mahalanobis:.3f}" if distance_mahalanobis > 0 else ""  # Ajout de la distance au titre.
    ax1.set_xlabel(f"PC1 ({variance_pc1 * 100:.1f} % de variance)", fontsize=12)  # Axe x avec pourcentage de variance.
    ax1.set_ylabel(f"PC2 ({variance_pc2 * 100:.1f} % de variance)", fontsize=12)  # Axe y avec pourcentage de variance.
    ax1.set_title(f"Graphe des individus ACP — Fake vs Réel{titre_dist}", fontsize=13)  # Titre de la figure.
    ax1.legend(loc="best", fontsize=10)  # Légende positionnée automatiquement.
    ax1.grid(True, alpha=0.3)  # Grille légère pour faciliter la lecture.
    plt.tight_layout()  # Ajuste les marges de la figure.
    chemin_fig1 = dossier_sortie / "graphe_individus_ellipses.png"  # Chemin de sauvegarde.
    fig1.savefig(chemin_fig1, dpi=200)  # Sauvegarde la figure en haute résolution.
    plt.close(fig1)  # Ferme la figure pour libérer la mémoire.
    print(f"Graphe des individus sauvegardé : {chemin_fig1}")  # Confirme la sauvegarde.

    # =====================================================================
    # FIGURE 2 : Cercle des corrélations (50 variables ou moins)
    # =====================================================================
    n_variables = X_train.shape[1]  # Nombre total de variables (pixels).
    MAX_VARS_CERCLE = 50  # Limite de variables affichées pour garder le graphe lisible.

    if n_variables <= MAX_VARS_CERCLE:  # Si le nombre de variables est raisonnable.
        indices_vars = np.arange(n_variables)  # On utilise toutes les variables.
    else:  # Sinon on sélectionne les 50 variables les plus contributives à PC1.
        contributions_pc1 = np.abs(acp.components_[0])  # Contributions absolues à PC1.
        indices_vars = np.argsort(contributions_pc1)[::-1][:MAX_VARS_CERCLE]  # Top 50 contributeurs.

    fig2, ax2 = plt.subplots(figsize=(8, 8))  # Crée la figure du cercle des corrélations.
    cercle = plt.Circle((0, 0), 1, color="lightgray", fill=False, linewidth=1.5)  # Cercle unité de référence.
    ax2.add_patch(cercle)  # Ajoute le cercle à l'axe.

    composantes = acp.components_[:2, :]  # Récupère les deux premières composantes de l'ACP.
    for i in indices_vars:  # Parcourt les variables sélectionnées.
        coord_x = composantes[0, i]  # Corrélation variable i avec PC1.
        coord_y = composantes[1, i]  # Corrélation variable i avec PC2.
        ax2.annotate(  # Affiche un label pour les variables les plus contributives.
            f"p{i}",  # Nom simplifié de la variable (pixel i).
            (coord_x, coord_y),  # Position de l'annotation.
            fontsize=5,  # Taille réduite pour éviter les chevauchements.
            ha="center",  # Centrage horizontal.
            color="#555555",  # Couleur gris foncé.
        )
        ax2.arrow(  # Trace une flèche depuis l'origine vers la variable.
            0, 0, coord_x, coord_y,  # De (0,0) vers (coord_x, coord_y).
            head_width=0.02,  # Taille de la tête de flèche.
            head_length=0.02,  # Longueur de la tête de flèche.
            fc="#3498db",  # Couleur de remplissage.
            ec="#2980b9",  # Couleur de contour.
            alpha=0.6,  # Légère transparence.
            length_includes_head=True,  # La longueur inclut la tête.
        )

    ax2.set_xlim(-1.15, 1.15)  # Borne l'axe x pour voir le cercle entier.
    ax2.set_ylim(-1.15, 1.15)  # Borne l'axe y pour voir le cercle entier.
    ax2.axhline(0, color="gray", linewidth=0.5)  # Ligne horizontale centrale.
    ax2.axvline(0, color="gray", linewidth=0.5)  # Ligne verticale centrale.
    ax2.set_aspect("equal")  # Force un repère orthonormé.
    ax2.set_xlabel(f"PC1 ({variance_pc1 * 100:.1f} %)", fontsize=11)  # Axe x.
    ax2.set_ylabel(f"PC2 ({variance_pc2 * 100:.1f} %)", fontsize=11)  # Axe y.
    titre_cercle = (
        f"Cercle des corrélations ACP\n"
        f"({len(indices_vars)} variables parmi {n_variables} — les plus contributives à PC1)"
        if n_variables > MAX_VARS_CERCLE
        else f"Cercle des corrélations ACP ({n_variables} variables)"
    )
    ax2.set_title(titre_cercle, fontsize=12)  # Titre de la figure.
    ax2.grid(True, alpha=0.2)  # Grille légère.
    plt.tight_layout()  # Ajuste les marges.
    chemin_fig2 = dossier_sortie / "cercle_correlations.png"  # Chemin de sauvegarde.
    fig2.savefig(chemin_fig2, dpi=200)  # Sauvegarde la figure.
    plt.close(fig2)  # Ferme la figure.
    print(f"Cercle des corrélations sauvegardé : {chemin_fig2}")  # Confirme la sauvegarde.

    # =====================================================================
    # FIGURE 3 : Barplot des 20 variables les plus contributives à PC1
    # =====================================================================
    contributions_abs = np.abs(acp.components_[0])  # Valeurs absolues des coordonnées sur PC1.
    top_indices = np.argsort(contributions_abs)[::-1][:20]  # Indices des 20 plus grandes contributions.
    top_valeurs = contributions_abs[top_indices]  # Valeurs des 20 plus grandes contributions.
    top_labels = [f"p{i}" for i in top_indices]  # Noms des variables (pixel_i).

    fig3, ax3 = plt.subplots(figsize=(12, 5))  # Crée la figure du barplot des contributions.
    barres = ax3.bar(  # Trace les barres.
        np.arange(20),  # Positions des barres.
        top_valeurs,  # Hauteurs (contributions à PC1).
        color="#e67e22",  # Couleur orange pour distinguer des autres figures.
        edgecolor="#d35400",  # Contour légèrement plus foncé.
        alpha=0.85,  # Légère transparence.
    )
    ax3.set_xticks(np.arange(20))  # Définit les positions des étiquettes.
    ax3.set_xticklabels(top_labels, rotation=45, ha="right", fontsize=9)  # Noms des variables.
    ax3.set_xlabel("Variable (pixel)", fontsize=11)  # Nom de l'axe x.
    ax3.set_ylabel("|Contribution à PC1|", fontsize=11)  # Nom de l'axe y.
    ax3.set_title("Top 20 variables les plus contributives à PC1\n(utiles pour identifier les zones discriminantes du visage)", fontsize=12)  # Titre.
    ax3.grid(True, axis="y", alpha=0.3)  # Grille horizontale uniquement.
    plt.tight_layout()  # Ajuste les marges.
    chemin_fig3 = dossier_sortie / "contributions_pc1.png"  # Chemin de sauvegarde.
    fig3.savefig(chemin_fig3, dpi=200)  # Sauvegarde la figure.
    plt.close(fig3)  # Ferme la figure.
    print(f"Contributions PC1 sauvegardées : {chemin_fig3}")  # Confirme la sauvegarde.

    # =====================================================================
    # COMPARAISON OPTIONNELLE : GaussianNB sur les composantes ACP
    # =====================================================================
    gnb = GaussianNB()  # Crée un classifieur bayésien gaussien.
    gnb.fit(X_pca, y_train)  # Entraîne le GaussianNB sur toutes les composantes ACP.
    y_pred_gnb = gnb.predict(X_pca)  # Prédit sur le jeu d'apprentissage (score indicatif).
    accuracy_gnb = float(accuracy_score(y_train, y_pred_gnb))  # Calcule l'accuracy du GaussianNB.
    print(f"[ACP Analyse] GaussianNB accuracy (train) : {accuracy_gnb:.4f}  —  Mahalanobis : {distance_mahalanobis:.3f}")  # Résumé console.

    return {  # Retourne les métriques calculées pour l'interface Streamlit.
        "distance_mahalanobis": distance_mahalanobis,  # Distance de Mahalanobis entre les deux classes.
        "variance_pc1": variance_pc1,  # Variance expliquée par PC1.
        "variance_pc2": variance_pc2,  # Variance expliquée par PC2.
        "n_composantes": n_composantes,  # Nombre de composantes retenues.
        "accuracy_gnb": accuracy_gnb,  # Accuracy du classifieur bayésien gaussien.
    }


def entrainer_et_evaluer(X: np.ndarray, y: np.ndarray) -> Pipeline:  # Déclare une fonction d'entraînement complet.
    """Sépare les données, entraîne ACP+SVM, évalue le modèle et affiche les résultats."""  # Explique le rôle.
    afficher_titre("Étape 2 - Séparation apprentissage / test")  # Affiche le titre de l'étape.
    X_train, X_test, y_train, y_test = train_test_split(  # Sépare X et y en deux parties.
        X,  # Données d'entrée.
        y,  # Labels associés.
        test_size=TAILLE_TEST,  # Proportion réservée au test.
        random_state=RANDOM_STATE,  # Graine pour reproduire le même découpage.
        stratify=y,  # Garde la même proportion de classes dans train et test.
    )  # Fin de train_test_split.

    print(f"Échantillons d'apprentissage : {len(X_train)}")  # Affiche le nombre d'exemples d'entraînement.
    print(f"Échantillons de test : {len(X_test)}")  # Affiche le nombre d'exemples de test.

    afficher_titre("Étape 2b - Analyse ACP approfondie (avant classification)")  # Affiche le titre de l'étape ACP exploratoire.
    # L'analyse ACP approfondie est lancée AVANT le pipeline de classification.
    # Elle sert à explorer visuellement les données et ne modifie pas l'entraînement SVM.
    metriques_acp = analyse_acp_approfondie(X_train, y_train, DOSSIER_SORTIE)  # Génère les trois figures ACP.
    print(f"  Variance PC1 : {metriques_acp['variance_pc1'] * 100:.1f}%  |  PC2 : {metriques_acp['variance_pc2'] * 100:.1f}%")  # Affiche les variances.
    print(f"  Distance de Mahalanobis entre classes : {metriques_acp['distance_mahalanobis']:.3f}")  # Affiche la distance.
    print(f"  Accuracy GaussianNB (indicative) : {metriques_acp['accuracy_gnb']:.4f}")  # Affiche l'accuracy bayésienne.

    afficher_titre("Étape 3 - Entraînement Standardisation + ACP + SVM")  # Affiche le titre de l'étape.
    modele = creer_pipeline_acp_svm()  # Crée le pipeline complet.
    modele.fit(X_train, y_train)  # Entraîne toutes les étapes sur les données d'apprentissage.

    acp = modele.named_steps["acp"]  # Récupère l'objet ACP entraîné.
    print(f"Dimension avant ACP : {X_train.shape[1]}")  # Affiche la dimension avant réduction.
    print(f"Nombre de composantes ACP retenues : {acp.n_components_}")  # Affiche le nombre d'axes retenus.
    print(f"Variance totale conservée : {np.sum(acp.explained_variance_ratio_) * 100:.2f}%")  # Affiche la variance gardée.

    afficher_titre("Étape 4 - Évaluation du modèle")  # Affiche le titre de l'étape.
    y_pred = modele.predict(X_test)  # Prédit les classes des exemples de test.
    exactitude = accuracy_score(y_test, y_pred)  # Calcule l'accuracy.
    print(f"Accuracy / Exactitude : {exactitude:.4f}")  # Affiche l'accuracy.
    print("\nRapport de classification :")  # Introduit le rapport.
    print(classification_report(y_test, y_pred, target_names=["Fake", "Reel"]))  # Affiche précision, rappel et F1-score.

    sauvegarder_figures(modele, y_test, y_pred)  # Sauvegarde les figures utiles pour le rapport.
    sauvegarder_modele(modele)  # Sauvegarde le modèle entraîné.
    return modele  # Retourne le modèle entraîné.


def sauvegarder_figures(modele: Pipeline, y_test: np.ndarray, y_pred: np.ndarray) -> None:  # Déclare une fonction qui sauvegarde les graphes.
    """Sauvegarde la matrice de confusion et la courbe de variance cumulée de l'ACP."""  # Explique le rôle.
    DOSSIER_SORTIE.mkdir(exist_ok=True)  # Crée le dossier de résultats s'il n'existe pas.

    matrice = confusion_matrix(y_test, y_pred, labels=[0, 1])  # Calcule la matrice de confusion dans l'ordre Fake, Réel.
    affichage = ConfusionMatrixDisplay(confusion_matrix=matrice, display_labels=["Fake", "Reel"])  # Prépare l'affichage.
    affichage.plot(values_format="d")  # Affiche la matrice avec des entiers.
    plt.title("Matrice de confusion - ACP + SVM")  # Ajoute un titre à la figure.
    plt.tight_layout()  # Ajuste les marges pour éviter les coupures.
    chemin_matrice = DOSSIER_SORTIE / "matrice_confusion.png"  # Chemin de sauvegarde de la matrice.
    plt.savefig(chemin_matrice, dpi=200)  # Sauvegarde la figure en bonne résolution.
    plt.close()  # Ferme la figure pour libérer la mémoire.
    print(f"Matrice de confusion sauvegardée : {chemin_matrice}")  # Affiche le chemin de sortie.

    acp = modele.named_steps["acp"]  # Récupère l'ACP entraînée.
    variance_cumulee = np.cumsum(acp.explained_variance_ratio_)  # Calcule la variance cumulée.
    plt.figure()  # Crée une nouvelle figure.
    plt.plot(np.arange(1, len(variance_cumulee) + 1), variance_cumulee)  # Trace la variance cumulée.
    plt.xlabel("Nombre de composantes ACP")  # Ajoute le nom de l'axe x.
    plt.ylabel("Variance cumulée")  # Ajoute le nom de l'axe y.
    plt.title("Variance conservée par l'ACP")  # Ajoute un titre à la figure.
    plt.grid(True)  # Ajoute une grille pour faciliter la lecture.
    plt.tight_layout()  # Ajuste les marges de la figure.
    chemin_variance = DOSSIER_SORTIE / "variance_acp.png"  # Chemin de sauvegarde de la courbe.
    plt.savefig(chemin_variance, dpi=200)  # Sauvegarde la figure.
    plt.close()  # Ferme la figure.
    print(f"Courbe de variance ACP sauvegardée : {chemin_variance}")  # Affiche le chemin de sortie.


def sauvegarder_modele(modele: Pipeline) -> None:  # Déclare une fonction de sauvegarde du modèle.
    """Sauvegarde le pipeline entraîné pour pouvoir prédire une nouvelle image plus tard."""  # Explique le rôle.
    DOSSIER_SORTIE.mkdir(exist_ok=True)  # Crée le dossier de sortie si nécessaire.
    contenu = {  # Crée un dictionnaire contenant le modèle et les paramètres importants.
        "modele": modele,  # Stocke le pipeline entraîné.
        "taille_image": TAILLE_IMAGE,  # Stocke la taille d'image utilisée.
        "labels_texte": LABELS_TEXTE,  # Stocke la correspondance 0/1 vers texte.
    }  # Fin du dictionnaire.
    joblib.dump(contenu, MODELE_SORTIE)  # Sauvegarde le dictionnaire dans un fichier joblib.
    print(f"Modèle sauvegardé : {MODELE_SORTIE}")  # Informe l'utilisateur du chemin de sauvegarde.


# =============================  # Séparateur visuel.
# 6) PRÉDICTION D'UNE IMAGE      # Fonction pour tester une image après entraînement.
# =============================  # Séparateur visuel.

def predire_image_unique(image_path: Path) -> None:  # Déclare une fonction qui prédit une image donnée.
    """Charge le modèle sauvegardé et prédit si une image est Réelle ou Fake."""  # Explique le rôle.
    if not MODELE_SORTIE.exists():  # Vérifie si le modèle a déjà été entraîné et sauvegardé.
        print("Le modèle n'existe pas encore. Lancez d'abord le programme sans --image pour entraîner.")  # Explique quoi faire.
        sys.exit(1)  # Arrête le programme.

    contenu = joblib.load(MODELE_SORTIE)  # Charge le fichier joblib.
    modele = contenu["modele"]  # Récupère le pipeline entraîné.
    detecteur = creer_detecteur_visage()  # Crée le détecteur de visage pour prétraiter la nouvelle image.
    image_bgr = cv2.imread(str(image_path))  # Lit l'image à prédire.
    if image_bgr is None:  # Vérifie si l'image n'a pas pu être lue.
        print(f"Impossible de lire l'image : {image_path}")  # Affiche une erreur claire.
        sys.exit(1)  # Arrête le programme.

    vecteur = preparer_image(image_bgr, detecteur)  # Prétraite l'image comme pendant l'entraînement.
    vecteur = vecteur.reshape(1, -1)  # Transforme le vecteur en matrice d'un seul échantillon.
    prediction = int(modele.predict(vecteur)[0])  # Prédit la classe 0 ou 1.
    texte = LABELS_TEXTE[prediction]  # Convertit la classe numérique en texte.
    print(f"Prédiction pour {image_path.name} : {texte}")  # Affiche le résultat final.


# 7) PROGRAMME PRINCIPAL         # Point d'entrée du fichier Python.

def analyser_arguments() -> argparse.Namespace:  # Déclare une fonction qui lit les options du terminal.
    """Analyse les arguments passés au programme depuis la ligne de commande."""  # Explique le rôle.
    parseur = argparse.ArgumentParser(description="TP ACP - Détection visage Réel ou Fake")  # Crée le parseur d'arguments.
    parseur.add_argument("--image", type=str, default=None, help="Chemin d'une image à prédire avec le modèle sauvegardé")  # Ajoute l'option --image.
    return parseur.parse_args()  # Retourne les arguments lus.


def main() -> None:  # Déclare la fonction principale du programme.
    """Exécute soit l'entraînement complet, soit la prédiction d'une seule image."""  # Explique le rôle.
    args = analyser_arguments()  # Lit les arguments fournis par l'utilisateur.
    DOSSIER_SORTIE.mkdir(exist_ok=True)  # Crée le dossier de résultats dès le début.

    if args.image is not None:  # Vérifie si l'utilisateur demande une prédiction directe.
        predire_image_unique(Path(args.image))  # Prédit l'image indiquée.
        return  # Termine le programme après la prédiction.

    afficher_titre("TP - Détection de visage Réel ou Fake avec ACP")  # Affiche le titre général du TP.
    print(f"Dossier du programme : {BASE_DIR}")  # Affiche le dossier du script.
    print(f"Dossier dataset utilisé : {DATASET_DIR}")  # Affiche le dossier scanné pour les données.
    print(f"Taille image utilisée : {TAILLE_IMAGE}")  # Affiche la taille d'image configurée.
    print(f"Variance ACP conservée : {N_COMPONENTS_ACP}")  # Affiche la proportion de variance visée.

    X, y, sources = construire_dataset(DATASET_DIR)  # Charge et construit la matrice des données.
    pd.DataFrame({"source": sources, "label": y}).to_csv(DOSSIER_SORTIE / "echantillons_utilises.csv", index=False)  # Sauvegarde les fichiers utilisés.
    entrainer_et_evaluer(X, y)  # Entraîne et évalue le modèle complet.

if __name__ == "__main__":
    main()