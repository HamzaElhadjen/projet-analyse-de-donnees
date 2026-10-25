# -*- coding: utf-8 -*-  # Indique que le fichier peut contenir des accents et du texte français.

"""  # Début de la documentation générale du fichier Streamlit.
Interface Streamlit pour le TP M1 IV : Détection de visage Réel ou Fake avec ACP/PCA.  # Résumé de l'objectif.

Ce fichier ne remplace pas le programme principal tp_acp_anti_spoofing.py.  # On garde la logique dans le fichier principal.
Il importe les fonctions déjà écrites dans tp_acp_anti_spoofing.py.  # Cela évite de recopier tout le code.
Il ajoute une interface graphique simple pour charger le dataset, entraîner le modèle et tester une image.  # Rôle de l'interface.

Organisation recommandée du dossier :  # Explication de l'organisation attendue.
    dossier_du_tp/  # Dossier principal du projet.
    ├── tp_acp_anti_spoofing.py  # Programme contenant la logique ACP + SVM.
    ├── app_streamlit_acp.py  # Interface Streamlit actuelle.
    ├── real/  # Dossier contenant les vrais visages.
    └── fake/  # Dossier contenant les visages falsifiés.

Commande d'exécution :  # Commande à lancer dans le terminal.
    streamlit run app_streamlit_acp.py  # Lance l'interface dans le navigateur.
"""  # Fin de la documentation générale.

from pathlib import Path  # Permet de manipuler proprement les chemins des fichiers et dossiers.
import io  # Sert à gérer les fichiers en mémoire, notamment pour les téléchargements.
import sys  # Sert à modifier le chemin d'importation Python si nécessaire.
import traceback  # Sert à afficher les erreurs détaillées dans l'interface en cas de problème.

import cv2  # OpenCV sert à lire les images et à préparer les images pour le modèle.
import joblib  # Sert à charger ou sauvegarder le modèle entraîné.
import matplotlib.pyplot as plt  # Sert à dessiner les figures affichées dans Streamlit.
import numpy as np  # Sert aux calculs numériques et à la manipulation de tableaux.
import pandas as pd  # Sert à construire des tableaux de résultats lisibles.
import streamlit as st  # Streamlit sert à créer l'interface graphique web.

from sklearn.metrics import accuracy_score  # Calcule le taux de bonnes prédictions.
from sklearn.metrics import classification_report  # Calcule précision, rappel et F1-score.
from sklearn.metrics import confusion_matrix  # Calcule la matrice de confusion.
from sklearn.metrics import ConfusionMatrixDisplay  # Affiche proprement la matrice de confusion.
from sklearn.model_selection import train_test_split  # Sépare les données en apprentissage et test.


# =============================  # Séparateur visuel.
# 1) CONFIGURATION DE BASE       # Paramètres généraux de l'application Streamlit.
# =============================  # Séparateur visuel.

BASE_DIR = Path(__file__).resolve().parent  # Dossier où se trouve ce fichier Streamlit.
RESULTATS_STREAMLIT = BASE_DIR / "resultats_streamlit_acp"  # Dossier où l'interface sauvegarde les résultats.
MODELE_STREAMLIT = RESULTATS_STREAMLIT / "modele_acp_anti_spoofing_streamlit.joblib"  # Chemin du modèle sauvegardé par l'interface.
NOM_MODULE_CORE = "tp_acp_anti_spoofing"  # Nom du fichier principal sans l'extension .py.

st.set_page_config(  # Configure l'apparence générale de la page Streamlit.
    page_title="TP ACP Anti-Spoofing",  # Titre affiché dans l'onglet du navigateur.
    layout="wide",  # Utilise toute la largeur de la page.
)  # Fin de la configuration de page.


# 2) CHARGEMENT DU CODE PRINCIPAL # Importation du fichier tp_acp_anti_spoofing.py.

@st.cache_resource(show_spinner=False)  # Met en cache l'import du module pour éviter de le recharger à chaque action.
def charger_module_core():  # Déclare une fonction qui importe le programme principal.
    """Importe le fichier tp_acp_anti_spoofing.py comme un module Python."""  # Explique le rôle de la fonction.
    if str(BASE_DIR) not in sys.path:  # Vérifie si le dossier courant est déjà dans les chemins Python.
        sys.path.insert(0, str(BASE_DIR))  # Ajoute le dossier courant pour permettre l'import local.
    try:  # Lance une tentative d'importation.
        module = __import__(NOM_MODULE_CORE)  # Importe le fichier tp_acp_anti_spoofing.py.
        return module  # Retourne le module importé.
    except Exception as erreur:  # Capture les erreurs d'importation.
        st.error("Impossible d'importer le fichier tp_acp_anti_spoofing.py.")  # Affiche un message d'erreur clair.
        st.code(str(erreur))  # Affiche l'erreur exacte.
        st.stop()  # Arrête l'application Streamlit pour éviter une suite incohérente.


core = charger_module_core()  # Importe le programme principal et le rend disponible dans toute l'interface.


# 3) FONCTIONS UTILITAIRES       # Fonctions utilisées par l'interface.

def appliquer_parametres_core(  # Déclare une fonction qui synchronise les paramètres de l'interface avec le code principal.
    dossier_dataset: Path,  # Dossier où se trouve le dataset.
    taille_image: tuple[int, int],  # Taille de redimensionnement des images.
    utiliser_detection_visage: bool,  # Indique si on utilise la détection du visage.
    variance_acp: float,  # Pourcentage de variance conservée par l'ACP.
    taille_test: float,  # Proportion des données utilisée pour le test.
    random_state: int,  # Graine aléatoire pour rendre les résultats reproductibles.
    frame_step_video: int,  # Intervalle d'extraction des frames vidéo.
    max_frames_par_video: int,  # Nombre maximal de frames extraites par vidéo.
    max_echantillons_total: int | None,  # Limite optionnelle du nombre total d'échantillons.
) -> None:  # La fonction ne retourne rien.
    """Applique les paramètres choisis dans Streamlit au module principal du TP."""  # Explique le rôle de la fonction.
    core.DATASET_DIR = dossier_dataset  # Définit le dossier du dataset utilisé par le programme principal.
    core.TAILLE_IMAGE = taille_image  # Définit la taille finale des images.
    core.UTILISER_DETECTION_VISAGE = utiliser_detection_visage  # Active ou désactive la détection de visage.
    core.N_COMPONENTS_ACP = variance_acp  # Définit la variance conservée par l'ACP.
    core.TAILLE_TEST = taille_test  # Définit la proportion de test.
    core.RANDOM_STATE = random_state  # Définit la graine aléatoire.
    core.FRAME_STEP_VIDEO = frame_step_video  # Définit la fréquence d'extraction vidéo.
    core.MAX_FRAMES_PAR_VIDEO = max_frames_par_video  # Définit le nombre maximal de frames par vidéo.
    core.MAX_ECHANTILLONS_TOTAL = max_echantillons_total  # Définit la limite optionnelle d'échantillons.
    core.DOSSIER_SORTIE = RESULTATS_STREAMLIT  # Redirige les sorties vers le dossier Streamlit.
    core.MODELE_SORTIE = MODELE_STREAMLIT  # Redirige la sauvegarde du modèle vers le fichier Streamlit.
    RESULTATS_STREAMLIT.mkdir(exist_ok=True)  # Crée le dossier de résultats s'il n'existe pas.


def entrainer_modele_interface(  # Déclare la fonction principale d'entraînement appelée par le bouton Streamlit.
    dossier_dataset: Path,  # Dossier contenant le dataset.
    taille_image: tuple[int, int],  # Taille finale des images.
    utiliser_detection_visage: bool,  # Activation ou non de la détection de visage.
    variance_acp: float,  # Variance conservée par l'ACP.
    taille_test: float,  # Proportion du jeu de test.
    random_state: int,  # Graine aléatoire.
    frame_step_video: int,  # Intervalle de frames pour les vidéos.
    max_frames_par_video: int,  # Maximum de frames par vidéo.
    max_echantillons_total: int | None,  # Limite optionnelle du nombre d'échantillons.
) -> dict:  # Retourne un dictionnaire contenant le modèle et les résultats.
    """Charge le dataset, applique l'ACP, entraîne le SVM, évalue le modèle et retourne les résultats."""  # Explique le rôle.
    appliquer_parametres_core(  # Applique les paramètres choisis dans l'interface.
        dossier_dataset=dossier_dataset,  # Transmet le dossier du dataset.
        taille_image=taille_image,  # Transmet la taille d'image.
        utiliser_detection_visage=utiliser_detection_visage,  # Transmet l'option de détection de visage.
        variance_acp=variance_acp,  # Transmet la variance ACP.
        taille_test=taille_test,  # Transmet la taille du test.
        random_state=random_state,  # Transmet la graine aléatoire.
        frame_step_video=frame_step_video,  # Transmet le pas d'extraction vidéo.
        max_frames_par_video=max_frames_par_video,  # Transmet le maximum de frames par vidéo.
        max_echantillons_total=max_echantillons_total,  # Transmet la limite d'échantillons.
    )  # Fin de l'appel d'application des paramètres.

    X, y, sources = core.construire_dataset(dossier_dataset)  # Construit la matrice X, les labels y et les sources des fichiers.

    if len(X) == 0:  # Vérifie si aucun échantillon n'a été chargé.
        raise ValueError("Aucun échantillon n'a été chargé depuis le dataset.")  # Arrête avec une erreur claire.

    classes, counts = np.unique(y, return_counts=True)  # Calcule les classes présentes et leurs effectifs.

    if len(classes) < 2:  # Vérifie si une seule classe est présente.
        raise ValueError("Le dataset doit contenir au moins deux classes : real et fake.")  # Arrête car la classification binaire est impossible.

    if np.min(counts) < 2:  # Vérifie si une classe contient moins de deux exemples.
        stratify = None  # Désactive la stratification si elle est impossible.
    else:  # Sinon, les deux classes sont assez représentées.
        stratify = y  # Active la stratification pour garder le même équilibre de classes dans train et test.

    X_train, X_test, y_train, y_test, sources_train, sources_test = train_test_split(  # Découpe le dataset en apprentissage et test.
        X,  # Données d'entrée.
        y,  # Labels des données.
        sources,  # Chemins des fichiers sources.
        test_size=taille_test,  # Proportion de test choisie dans l'interface.
        random_state=random_state,  # Graine aléatoire choisie.
        stratify=stratify,  # Stratification si possible.
    )  # Fin du découpage train/test.

    modele = core.creer_pipeline_acp_svm()  # Crée le pipeline StandardScaler + ACP + SVM du programme principal.

    # --- Analyse ACP approfondie AVANT le fit() du pipeline ---
    # On appelle ici la fonction d'analyse qui génère trois figures exploratoires
    # dans le dossier de résultats. Ce n'est pas la même ACP que le pipeline :
    # c'est une ACP réalisée uniquement pour la visualisation et l'exploration.
    try:
        metriques_acp = core.analyse_acp_approfondie(X_train, y_train, RESULTATS_STREAMLIT)  # Génère les figures ACP.
    except Exception as erreur_acp:
        metriques_acp = {}  # Si l'analyse échoue, on continue sans bloquer l'entraînement.
        import traceback as tb  # Import local pour afficher l'erreur sans arrêter l'interface.
        print(f"Avertissement analyse ACP : {erreur_acp}")  # Affiche l'avertissement dans la console.

    modele.fit(X_train, y_train)  # Entraîne le pipeline sur les données d'apprentissage.

    y_pred = modele.predict(X_test)  # Prédit les classes des données de test.
    accuracy = accuracy_score(y_test, y_pred)  # Calcule l'accuracy du modèle.
    rapport_dict = classification_report(  # Calcule le rapport de classification sous forme de dictionnaire.
        y_test,  # Labels réels.
        y_pred,  # Labels prédits.
        target_names=["Fake", "Reel"],  # Noms des classes dans l'ordre 0 puis 1.
        output_dict=True,  # Demande une sortie exploitable par pandas.
        zero_division=0,  # Évite les erreurs si une métrique est impossible à calculer.
    )  # Fin du calcul du rapport.
    rapport_texte = classification_report(  # Calcule aussi une version texte du rapport.
        y_test,  # Labels réels.
        y_pred,  # Labels prédits.
        target_names=["Fake", "Reel"],  # Noms des classes.
        zero_division=0,  # Évite les divisions par zéro.
    )  # Fin du rapport texte.
    matrice = confusion_matrix(y_test, y_pred, labels=[0, 1])  # Calcule la matrice de confusion dans l'ordre Fake/Reel.

    core.sauvegarder_figures(modele, y_test, y_pred)  # Sauvegarde les figures dans le dossier de résultats.
    core.sauvegarder_modele(modele)  # Sauvegarde le modèle entraîné.

    df_sources = pd.DataFrame(  # Crée un tableau des échantillons utilisés.
        {  # Début du dictionnaire de colonnes.
            "source": sources,  # Chemin ou nom de chaque fichier utilisé.
            "label": y,  # Label numérique de chaque échantillon.
            "classe": [core.LABELS_TEXTE[int(label)] for label in y],  # Label textuel de chaque échantillon.
        }  # Fin du dictionnaire.
    )  # Fin de la création du DataFrame.
    df_sources.to_csv(RESULTATS_STREAMLIT / "echantillons_utilises_streamlit.csv", index=False)  # Sauvegarde les échantillons utilisés.

    return {  # Retourne tous les éléments utiles à l'interface.
        "modele": modele,  # Modèle entraîné.
        "X_shape": X.shape,  # Dimension globale du dataset vectorisé.
        "y": y,  # Labels complets.
        "sources": sources,  # Sources complètes.
        "X_train_shape": X_train.shape,  # Dimensions de l'ensemble d'apprentissage.
        "X_test_shape": X_test.shape,  # Dimensions de l'ensemble de test.
        "y_test": y_test,  # Labels réels du test.
        "y_pred": y_pred,  # Labels prédits du test.
        "sources_test": sources_test,  # Sources des fichiers de test.
        "accuracy": accuracy,  # Accuracy finale.
        "rapport_dict": rapport_dict,  # Rapport de classification en dictionnaire.
        "rapport_texte": rapport_texte,  # Rapport de classification en texte.
        "matrice": matrice,  # Matrice de confusion.
        "df_sources": df_sources,  # Tableau des sources utilisées.
        "metriques_acp": metriques_acp,  # Métriques de l'analyse ACP approfondie.
    }  # Fin du dictionnaire retourné.


def afficher_matrice_confusion(matrice: np.ndarray) -> None:  # Déclare une fonction d'affichage de la matrice de confusion.
    """Affiche la matrice de confusion dans l'interface Streamlit."""  # Explique le rôle de la fonction.
    fig, ax = plt.subplots(figsize=(5, 4))  # Crée une figure Matplotlib.
    affichage = ConfusionMatrixDisplay(confusion_matrix=matrice, display_labels=["Fake", "Reel"])  # Prépare l'affichage.
    affichage.plot(values_format="d", ax=ax)  # Dessine la matrice avec des valeurs entières.
    ax.set_title("Matrice de confusion")  # Ajoute un titre à la figure.
    st.pyplot(fig)  # Affiche la figure dans Streamlit.
    plt.close(fig)  # Ferme la figure pour libérer la mémoire.


def afficher_variance_acp(modele) -> None:  # Déclare une fonction qui affiche la variance cumulée de l'ACP.
    """Affiche la courbe de variance cumulée conservée par l'ACP."""  # Explique le rôle de la fonction.
    acp = modele.named_steps["acp"]  # Récupère l'objet ACP/PCA depuis le pipeline.
    variance_cumulee = np.cumsum(acp.explained_variance_ratio_)  # Calcule la variance cumulée.
    fig, ax = plt.subplots(figsize=(7, 4))  # Crée une figure Matplotlib.
    ax.plot(np.arange(1, len(variance_cumulee) + 1), variance_cumulee)  # Trace la variance cumulée.
    ax.set_xlabel("Nombre de composantes ACP")  # Définit le nom de l'axe horizontal.
    ax.set_ylabel("Variance cumulée")  # Définit le nom de l'axe vertical.
    ax.set_title("Variance conservée par l'ACP")  # Ajoute le titre du graphique.
    ax.grid(True)  # Affiche une grille pour faciliter la lecture.
    st.pyplot(fig)  # Affiche la figure dans Streamlit.
    plt.close(fig)  # Ferme la figure pour libérer la mémoire.


def lire_image_upload(uploaded_file) -> np.ndarray | None:  # Déclare une fonction qui lit une image envoyée dans Streamlit.
    """Convertit un fichier image uploadé par l'utilisateur en image OpenCV BGR."""  # Explique le rôle.
    if uploaded_file is None:  # Vérifie si aucun fichier n'a été envoyé.
        return None  # Retourne None car il n'y a rien à lire.
    bytes_data = uploaded_file.getvalue()  # Récupère le contenu binaire du fichier uploadé.
    tableau = np.frombuffer(bytes_data, np.uint8)  # Convertit les octets en tableau NumPy.
    image_bgr = cv2.imdecode(tableau, cv2.IMREAD_COLOR)  # Décode l'image au format OpenCV BGR.
    return image_bgr  # Retourne l'image décodée.


def predire_image_interface(modele, image_bgr: np.ndarray) -> tuple[str, int]:  # Déclare une fonction qui prédit une image uploadée.
    """Prétraite une image puis prédit si elle appartient à la classe Réel ou Fake."""  # Explique le rôle.
    detecteur = core.creer_detecteur_visage()  # Crée le détecteur de visage utilisé par le code principal.
    vecteur = core.preparer_image(image_bgr, detecteur)  # Prétraite l'image comme pendant l'entraînement.
    vecteur = vecteur.reshape(1, -1)  # Transforme le vecteur en matrice contenant un seul exemple.
    prediction = int(modele.predict(vecteur)[0])  # Prédit la classe numérique 0 ou 1.
    texte = core.LABELS_TEXTE[prediction]  # Convertit la classe numérique en texte.
    return texte, prediction  # Retourne le texte et le label numérique.


def charger_modele_sauvegarde():  # Déclare une fonction qui charge un modèle déjà entraîné.
    """Charge le modèle sauvegardé par l'interface s'il existe."""  # Explique le rôle.
    if not MODELE_STREAMLIT.exists():  # Vérifie si le fichier modèle existe.
        return None  # Retourne None si aucun modèle n'est sauvegardé.
    contenu = joblib.load(MODELE_STREAMLIT)  # Charge le fichier joblib.
    return contenu.get("modele")  # Retourne le pipeline entraîné.


def dataframe_rapport(rapport_dict: dict) -> pd.DataFrame:  # Déclare une fonction qui transforme le rapport sklearn en tableau lisible.
    """Transforme le rapport de classification en DataFrame lisible dans Streamlit."""  # Explique le rôle.
    df = pd.DataFrame(rapport_dict).transpose()  # Transforme le dictionnaire en tableau pandas.
    return df.round(4)  # Arrondit les valeurs pour un affichage propre.


# 4) BARRE LATÉRALE              # Paramètres réglables par l'utilisateur.

st.sidebar.title(" Paramètres")  # Affiche le titre de la barre latérale.

chemin_dataset_texte = st.sidebar.text_input(  # Crée un champ texte pour le dossier du dataset.
    "Dossier du dataset",  # Libellé affiché à l'utilisateur.
    value=str(BASE_DIR),  # Valeur par défaut : même dossier que l'application.
    help="Par défaut, le dataset est supposé être dans le même dossier que app_streamlit_acp.py.",  # Aide affichée au survol.
)  # Fin du champ texte.

dossier_dataset = Path(chemin_dataset_texte).expanduser().resolve()  # Convertit le texte en chemin absolu.

largeur_image = st.sidebar.slider(  # Crée un slider pour la largeur de l'image.
    "Largeur image",  # Nom du paramètre.
    min_value=48,  # Valeur minimale.
    max_value=160,  # Valeur maximale.
    value=96,  # Valeur par défaut.
    step=16,  # Pas du slider.
)  # Fin du slider.

hauteur_image = st.sidebar.slider(  # Crée un slider pour la hauteur de l'image.
    "Hauteur image",  # Nom du paramètre.
    min_value=48,  # Valeur minimale.
    max_value=160,  # Valeur maximale.
    value=96,  # Valeur par défaut.
    step=16,  # Pas du slider.
)  # Fin du slider.

variance_acp = st.sidebar.slider(  # Crée un slider pour la variance ACP conservée.
    "Variance conservée par ACP",  # Nom du paramètre.
    min_value=0.80,  # Minimum possible.
    max_value=0.99,  # Maximum possible.
    value=0.95,  # Valeur recommandée.
    step=0.01,  # Pas de modification.
)  # Fin du slider.

taille_test = st.sidebar.slider(  # Crée un slider pour la proportion de test.
    "Partie test",  # Nom du paramètre.
    min_value=0.10,  # Minimum possible.
    max_value=0.40,  # Maximum possible.
    value=0.25,  # Valeur par défaut.
    step=0.05,  # Pas de modification.
)  # Fin du slider.

utiliser_detection_visage = st.sidebar.checkbox(  # Crée une case à cocher pour la détection du visage.
    "Utiliser la détection de visage",  # Texte affiché à l'utilisateur.
    value=True,  # Option activée par défaut.
)  # Fin de la case à cocher.

frame_step_video = st.sidebar.slider(  # Crée un slider pour le pas d'extraction vidéo.
    "Pas d'extraction vidéo",  # Nom du paramètre.
    min_value=5,  # Minimum possible.
    max_value=60,  # Maximum possible.
    value=15,  # Valeur par défaut.
    step=5,  # Pas de modification.
)  # Fin du slider.

max_frames_par_video = st.sidebar.slider(  # Crée un slider pour le nombre maximal de frames par vidéo.
    "Max frames par vidéo",  # Nom du paramètre.
    min_value=5,  # Minimum possible.
    max_value=100,  # Maximum possible.
    value=40,  # Valeur par défaut.
    step=5,  # Pas de modification.
)  # Fin du slider.

limiter_dataset = st.sidebar.checkbox(  # Crée une option pour limiter le dataset pendant les tests rapides.
    "Limiter le nombre d'échantillons",  # Texte affiché.
    value=False,  # Désactivé par défaut.
)  # Fin de la case à cocher.

if limiter_dataset:  # Vérifie si l'utilisateur veut limiter le dataset.
    max_echantillons_total = st.sidebar.number_input(  # Demande le nombre maximal d'échantillons.
        "Nombre maximal d'échantillons",  # Libellé affiché.
        min_value=50,  # Minimum accepté.
        max_value=10000,  # Maximum accepté.
        value=1000,  # Valeur par défaut.
        step=50,  # Pas d'incrémentation.
    )  # Fin du champ numérique.
    max_echantillons_total = int(max_echantillons_total)  # Convertit la valeur en entier Python.
else:  # Si l'utilisateur ne limite pas le dataset.
    max_echantillons_total = None  # Aucune limite n'est appliquée.

random_state = st.sidebar.number_input(  # Crée un champ pour la graine aléatoire.
    "Random state",  # Nom du paramètre.
    min_value=0,  # Minimum.
    max_value=9999,  # Maximum.
    value=42,  # Valeur par défaut.
    step=1,  # Pas d'incrémentation.
)  # Fin du champ numérique.
random_state = int(random_state)  # Convertit la graine en entier.

# 5) INTERFACE PRINCIPALE        # Contenu visible dans la page principale.

st.title(" TP ACP — Détection de visage Réel ou Fake")  # Affiche le grand titre de l'application.
st.write(  # Affiche une courte description du TP.
    "Cette interface applique le pipeline : **Dataset → Prétraitement → ACP/PCA → SVM → Évaluation → Prédiction**."  # Texte explicatif.
)  # Fin du texte.

st.info(  # Affiche une information importante dans un encadré.
    "Le dataset doit contenir deux classes : `real` pour les vrais visages et `fake` pour les attaques. "  # Première partie du message.
    "Le dataset peut contenir des images ou des vidéos."  # Deuxième partie du message.
)  # Fin de l'encadré.

tab_dataset, tab_train, tab_acp, tab_prediction, tab_explication = st.tabs(  # Crée cinq onglets principaux.
    [" Dataset", " Entraînement", " Analyse ACP", " Prédiction", " Explication"]  # Noms des onglets.
)  # Fin de la création des onglets.

# 6) ONGLET DATASET              # Vérification du dossier dataset.

with tab_dataset:  # Début de l'onglet Dataset.
    st.subheader(" Vérification du dataset")  # Affiche le sous-titre.
    st.write(f"Dossier actuellement sélectionné : `{dossier_dataset}`")  # Affiche le dossier choisi.

    if dossier_dataset.exists():  # Vérifie si le dossier existe.
        st.success("Le dossier existe.")  # Confirme que le dossier est trouvé.
    else:  # Sinon le dossier n'existe pas.
        st.error("Le dossier indiqué n'existe pas.")  # Affiche une erreur.

    st.markdown("### Organisation recommandée")  # Affiche un titre pour l'organisation.
    st.code(  # Affiche l'exemple d'organisation en bloc de code.
        """mon_tp/
├── tp_acp_anti_spoofing.py
├── app_streamlit_acp.py
├── real/
│   ├── image1.jpg
│   └── video1.avi
└── fake/
    ├── image2.jpg
    └── video2.avi""",  # Exemple de structure de fichiers.
        language="text",  # Langage du bloc.
    )  # Fin du bloc de code.

    if st.button("Scanner rapidement le dataset", type="secondary"):  # Crée un bouton pour scanner le dataset.
        try:  # Lance un essai pour éviter que l'interface se ferme en cas d'erreur.
            appliquer_parametres_core(  # Applique les paramètres avant le scan.
                dossier_dataset=dossier_dataset,  # Dossier choisi.
                taille_image=(largeur_image, hauteur_image),  # Taille d'image choisie.
                utiliser_detection_visage=utiliser_detection_visage,  # Option de détection.
                variance_acp=variance_acp,  # Variance ACP.
                taille_test=taille_test,  # Taille de test.
                random_state=random_state,  # Graine aléatoire.
                frame_step_video=frame_step_video,  # Pas vidéo.
                max_frames_par_video=max_frames_par_video,  # Max frames vidéo.
                max_echantillons_total=max_echantillons_total,  # Limite d'échantillons.
            )  # Fin de l'application des paramètres.
            with st.spinner("Chargement et prétraitement du dataset..."):  # Affiche un spinner pendant le chargement.
                X_scan, y_scan, sources_scan = core.construire_dataset(dossier_dataset)  # Construit rapidement le dataset.
            st.success("Dataset chargé avec succès.")  # Affiche un message de succès.
            col1, col2, col3 = st.columns(3)  # Crée trois colonnes pour les métriques.
            col1.metric("Nombre d'échantillons", len(y_scan))  # Affiche le nombre d'échantillons.
            col2.metric("Dimension d'un vecteur", X_scan.shape[1] if len(X_scan) else 0)  # Affiche la dimension des vecteurs.
            col3.metric("Classes trouvées", len(np.unique(y_scan)))  # Affiche le nombre de classes.
            df_counts = pd.DataFrame(  # Crée un tableau des effectifs par classe.
                {  # Début du dictionnaire.
                    "classe": [core.LABELS_TEXTE[int(c)] for c in np.unique(y_scan)],  # Noms des classes.
                    "nombre": [int(np.sum(y_scan == c)) for c in np.unique(y_scan)],  # Nombre d'exemples par classe.
                }  # Fin du dictionnaire.
            )  # Fin de la création du DataFrame.
            st.dataframe(df_counts, use_container_width=True)  # Affiche le tableau des classes.
            st.markdown("### Premiers fichiers utilisés")  # Titre pour les exemples de fichiers.
            st.dataframe(pd.DataFrame({"source": sources_scan[:20], "label": y_scan[:20]}), use_container_width=True)  # Affiche les 20 premiers fichiers.
        except Exception as erreur:  # Capture une erreur éventuelle.
            st.error("Erreur pendant le scan du dataset.")  # Affiche une erreur claire.
            st.code("".join(traceback.format_exception_only(type(erreur), erreur)))  # Affiche le résumé de l'erreur.


# 7) ONGLET ENTRAÎNEMENT         # Lancement de l'apprentissage et résultats.

with tab_train:  # Début de l'onglet entraînement.
    st.subheader(" Entraîner le modèle ACP + SVM")  # Affiche le sous-titre.
    st.write("Clique sur le bouton ci-dessous pour lancer tout le pipeline du TP.")  # Explique l'action.

    col_param1, col_param2, col_param3, col_param4 = st.columns(4)  # Crée quatre colonnes de paramètres résumés.
    col_param1.metric("Taille image", f"{largeur_image}×{hauteur_image}")  # Affiche la taille d'image.
    col_param2.metric("Variance ACP", f"{variance_acp:.2f}")  # Affiche la variance ACP.
    col_param3.metric("Test", f"{taille_test:.0%}")  # Affiche la proportion de test.
    col_param4.metric("Détection visage", "Oui" if utiliser_detection_visage else "Non")  # Affiche l'état de la détection.

    bouton_train = st.button("Lancer l'entraînement", type="primary")  # Crée le bouton principal d'entraînement.

    if bouton_train:  # Vérifie si l'utilisateur a cliqué sur le bouton.
        try:  # Lance l'entraînement dans un bloc de sécurité.
            with st.spinner("Entraînement en cours : chargement, ACP, SVM et évaluation..."):  # Affiche une animation pendant l'entraînement.
                resultats = entrainer_modele_interface(  # Appelle la fonction d'entraînement.
                    dossier_dataset=dossier_dataset,  # Dossier du dataset.
                    taille_image=(largeur_image, hauteur_image),  # Taille image.
                    utiliser_detection_visage=utiliser_detection_visage,  # Option détection.
                    variance_acp=variance_acp,  # Variance ACP.
                    taille_test=taille_test,  # Partie test.
                    random_state=random_state,  # Graine aléatoire.
                    frame_step_video=frame_step_video,  # Pas vidéo.
                    max_frames_par_video=max_frames_par_video,  # Frames max.
                    max_echantillons_total=max_echantillons_total,  # Limite dataset.
                )  # Fin de l'appel d'entraînement.
            st.session_state["resultats"] = resultats  # Sauvegarde les résultats dans la session Streamlit.
            st.session_state["modele"] = resultats["modele"]  # Sauvegarde le modèle entraîné dans la session.
            st.success("Modèle entraîné et sauvegardé avec succès.")  # Affiche un message de succès.
        except Exception as erreur:  # Capture les erreurs d'entraînement.
            st.error("Erreur pendant l'entraînement.")  # Affiche un message d'erreur.
            st.code("".join(traceback.format_exception(type(erreur), erreur, erreur.__traceback__)))  # Affiche l'erreur complète pour débogage.

    if "resultats" in st.session_state:  # Vérifie si des résultats existent déjà.
        resultats = st.session_state["resultats"]  # Récupère les résultats stockés.
        modele = resultats["modele"]  # Récupère le modèle entraîné.
        acp = modele.named_steps["acp"]  # Récupère l'objet ACP depuis le pipeline.

        st.markdown("### Résultats principaux")  # Affiche un titre de section.
        col1, col2, col3, col4 = st.columns(4)  # Crée quatre colonnes pour les métriques.
        col1.metric("Accuracy", f"{resultats['accuracy']:.4f}")  # Affiche l'accuracy.
        col2.metric("Échantillons", resultats["X_shape"][0])  # Affiche le nombre total d'échantillons.
        col3.metric("Dimension avant ACP", resultats["X_shape"][1])  # Affiche la dimension initiale.
        col4.metric("Composantes ACP", acp.n_components_)  # Affiche le nombre de composantes retenues.

        st.markdown("### Rapport de classification")  # Affiche un titre pour les métriques détaillées.
        st.dataframe(dataframe_rapport(resultats["rapport_dict"]), use_container_width=True)  # Affiche le rapport sous forme de tableau.

        col_graph1, col_graph2 = st.columns(2)  # Crée deux colonnes pour les graphiques.
        with col_graph1:  # Début de la première colonne.
            afficher_matrice_confusion(resultats["matrice"])  # Affiche la matrice de confusion.
        with col_graph2:  # Début de la deuxième colonne.
            afficher_variance_acp(modele)  # Affiche la variance cumulée ACP.

        st.markdown("### Fichiers générés")  # Affiche un titre pour les fichiers de sortie.
        st.write(f"Modèle sauvegardé : `{MODELE_STREAMLIT}`")  # Affiche le chemin du modèle sauvegardé.
        st.write(f"Dossier des résultats : `{RESULTATS_STREAMLIT}`")  # Affiche le dossier des résultats.

        if MODELE_STREAMLIT.exists():  # Vérifie si le modèle sauvegardé existe.
            with open(MODELE_STREAMLIT, "rb") as fichier_modele:  # Ouvre le modèle en lecture binaire.
                st.download_button(  # Crée un bouton de téléchargement du modèle.
                    "Télécharger le modèle entraîné",  # Texte du bouton.
                    data=fichier_modele.read(),  # Données du modèle.
                    file_name="modele_acp_anti_spoofing_streamlit.joblib",  # Nom proposé au téléchargement.
                    mime="application/octet-stream",  # Type MIME générique.
                )  # Fin du bouton de téléchargement.


# 7b) ONGLET ANALYSE ACP         # Graphes exploratoires de l'ACP (graphe individus, cercle, contributions).

with tab_acp:  # Début de l'onglet Analyse ACP.
    st.subheader(" Analyse en Composantes Principales")  # Titre de l'onglet.
    st.write(
        "Ces graphiques sont générés **avant la classification SVM**, à partir des données d'apprentissage. "
        "Ils permettent d'explorer visuellement la structure des données et de comprendre ce que l'ACP capture."
    )  # Explique la finalité de cet onglet.

    if "resultats" not in st.session_state:  # Vérifie si l'entraînement a été lancé.
        st.info("Lance d'abord l'entraînement dans l'onglet **Entraînement** pour voir les graphiques ACP.")  # Invite à entraîner.
    else:  # Si des résultats existent dans la session.
        resultats = st.session_state["resultats"]  # Récupère les résultats.
        metriques_acp = resultats.get("metriques_acp", {})  # Récupère les métriques ACP si elles existent.

        # --- Métriques résumées ---
        if metriques_acp:  # Vérifie si les métriques ont bien été calculées.
            st.markdown("### Métriques ACP exploratoires")  # Titre de section.
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)  # Crée quatre colonnes de métriques.
            col_m1.metric("Variance PC1", f"{metriques_acp.get('variance_pc1', 0) * 100:.1f} %")  # Variance PC1.
            col_m2.metric("Variance PC2", f"{metriques_acp.get('variance_pc2', 0) * 100:.1f} %")  # Variance PC2.
            col_m3.metric(
                "Distance Mahalanobis",
                f"{metriques_acp.get('distance_mahalanobis', 0):.3f}",
                help="Plus la distance est élevée, plus les classes sont séparées dans l'espace ACP.",
            )  # Distance de Mahalanobis entre les deux classes.
            col_m4.metric(
                "Accuracy GaussianNB",
                f"{metriques_acp.get('accuracy_gnb', 0):.4f}",
                help="Accuracy d'un classifieur bayésien gaussien entraîné sur les composantes ACP (indicatif).",
            )  # Accuracy du classifieur bayésien sur les données ACP.

        # --- Figure 1 : Graphe des individus avec ellipses ---
        chemin_individus = RESULTATS_STREAMLIT / "graphe_individus_ellipses.png"  # Chemin de la figure 1.
        with st.expander(" Graphe des individus (PC1 vs PC2) avec ellipses de confiance", expanded=True):  # Expandeur pour regrouper.
            if chemin_individus.exists():  # Vérifie que la figure a bien été créée.
                st.image(str(chemin_individus), caption="Projection des individus sur PC1 et PC2. Les ellipses représentent l'intervalle de confiance à 95 % sous hypothèse de normalité bivariée.", use_container_width=True)  # Affiche l'image.
                st.caption(
                    "**Comment lire ce graphe ?**  \n"
                    "- Les points bleus/triangles représentent les visages **réels**, les rouges/ronds les **fakes**.  \n"
                    "- Si les deux nuages de points sont bien séparés sur PC1 ou PC2, l'ACP aide la classification.  \n"
                    "- Les ellipses indiquent la zone couvrant ~95 % des individus de chaque classe (hypothèse normale).  \n"
                    "- Un **chevauchement important** des ellipses suggère une séparation difficile et des erreurs SVM possibles.  \n"
                    "- La **distance de Mahalanobis** mesure l'écart entre les deux centres tenant compte des covariances."
                )  # Explications pédagogiques sous la figure.
            else:  # Si la figure n'a pas été générée.
                st.warning("Figure non disponible. Vérifiez que l'entraînement s'est terminé sans erreur.")  # Message d'avertissement.

        # --- Figure 2 : Cercle des corrélations ---
        chemin_cercle = RESULTATS_STREAMLIT / "cercle_correlations.png"  # Chemin de la figure 2.
        with st.expander(" Cercle des corrélations", expanded=False):  # Expandeur réduit par défaut.
            if chemin_cercle.exists():  # Vérifie que la figure existe.
                st.image(str(chemin_cercle), caption="Cercle des corrélations : position des variables dans le plan PC1-PC2.", use_container_width=True)  # Affiche l'image.
                st.caption(
                    "**Comment lire ce graphe ?**  \n"
                    "- Chaque flèche représente un pixel (variable) projeté dans le plan PC1-PC2.  \n"
                    "- Les flèches proches du **cercle unité** (bord extérieur) sont bien représentées par les deux premières composantes.  \n"
                    "- Des flèches dans la **même direction** indiquent des pixels très corrélés entre eux.  \n"
                    "- Les pixels orientés vers **PC1 (horizontalement)** contribuent le plus à discriminer les classes.  \n"
                    "- Des flèches courtes (proches du centre) indiquent des variables peu informatives pour les deux premières composantes."
                )  # Explications pédagogiques.
            else:  # Si la figure n'existe pas.
                st.warning("Figure non disponible.")  # Message d'avertissement.

        # --- Figure 3 : Barplot des contributions à PC1 ---
        chemin_contrib = RESULTATS_STREAMLIT / "contributions_pc1.png"  # Chemin de la figure 3.
        with st.expander(" Top 20 variables les plus contributives à PC1", expanded=False):  # Expandeur réduit.
            if chemin_contrib.exists():  # Vérifie que la figure existe.
                st.image(str(chemin_contrib), caption="Les 20 pixels ayant le plus fort poids absolu sur la première composante principale.", use_container_width=True)  # Affiche l'image.
                st.caption(
                    "**Comment lire ce graphe ?**  \n"
                    "- Les barres les plus hautes représentent les pixels qui **définissent le plus** la direction PC1.  \n"
                    "- En anti-spoofing, les pixels discriminants correspondent souvent aux **contours du visage**, aux **zones de texture** "
                    "(peau, yeux, front) ou aux **artefacts de compression** présents dans les photos/vidéos falsifiées.  \n"
                    "- Si les indices de pixels sont groupés (ex : pixels 200–250 consécutifs), cela peut indiquer "
                    "une **zone spatiale précise** de l'image (bas du visage, bord de l'image, etc.)."
                )  # Explications pédagogiques.
            else:  # Si la figure n'existe pas.
                st.warning("Figure non disponible.")  # Message d'avertissement.

        st.markdown("### Interprétation globale")  # Section d'interprétation.
        st.info(
            "L'ACP réalisée ici est **exploratoire** : elle précède la classification et ne sert pas uniquement "
            "à réduire la dimension.  \n"
            "Elle permet de vérifier que les données contiennent bien de l'information discriminante avant "
            "d'entraîner le SVM, et d'identifier les zones du visage les plus importantes pour la détection d'attaques."
        )  # Conclusion pédagogique.


# 8) ONGLET PRÉDICTION           # Test d'une image avec le modèle entraîné.

with tab_prediction:  # Début de l'onglet prédiction.
    st.subheader(" Tester une image")  # Affiche le sous-titre.
    st.write("Après l'entraînement, envoie une image pour savoir si elle est classée comme Réel ou Fake.")  # Explique le fonctionnement.

    modele_disponible = st.session_state.get("modele")  # Essaie de récupérer le modèle de la session.

    if modele_disponible is None:  # Vérifie si aucun modèle n'est dans la session.
        modele_disponible = charger_modele_sauvegarde()  # Essaie de charger un modèle sauvegardé.

    if modele_disponible is None:  # Vérifie si aucun modèle n'est disponible.
        st.warning("Aucun modèle entraîné n'est disponible. Lance d'abord l'entraînement.")  # Demande d'entraîner d'abord.
    else:  # Si un modèle est disponible.
        uploaded_image = st.file_uploader(  # Crée un composant d'upload d'image.
            "Choisis une image de visage",  # Texte affiché.
            type=["jpg", "jpeg", "png", "bmp", "webp"],  # Extensions acceptées.
        )  # Fin de l'upload.

        image_bgr = lire_image_upload(uploaded_image)  # Lit l'image uploadée au format OpenCV.

        if image_bgr is not None:  # Vérifie si une image a été correctement lue.
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Convertit BGR vers RGB pour l'affichage.
            st.image(image_rgb, caption="Image envoyée", use_container_width=True)  # Affiche l'image dans Streamlit.

            if st.button("Prédire cette image", type="primary"):  # Crée un bouton de prédiction.
                try:  # Lance la prédiction dans un bloc sécurisé.
                    appliquer_parametres_core(  # Applique les paramètres actuels avant le prétraitement.
                        dossier_dataset=dossier_dataset,  # Dossier dataset.
                        taille_image=(largeur_image, hauteur_image),  # Taille image.
                        utiliser_detection_visage=utiliser_detection_visage,  # Détection visage.
                        variance_acp=variance_acp,  # Variance ACP.
                        taille_test=taille_test,  # Taille test.
                        random_state=random_state,  # Graine.
                        frame_step_video=frame_step_video,  # Pas vidéo.
                        max_frames_par_video=max_frames_par_video,  # Max frames.
                        max_echantillons_total=max_echantillons_total,  # Limite.
                    )  # Fin de l'application des paramètres.
                    texte_prediction, label_prediction = predire_image_interface(modele_disponible, image_bgr)  # Prédit la classe.
                    if label_prediction == 1:  # Vérifie si la prédiction est la classe réelle.
                        st.success(f"Résultat : {texte_prediction} ")  # Affiche un résultat positif.
                    else:  # Sinon la prédiction est fake.
                        st.error(f"Résultat : {texte_prediction} ")  # Affiche un résultat d'attaque.
                except Exception as erreur:  # Capture les erreurs de prédiction.
                    st.error("Erreur pendant la prédiction.")  # Affiche un message d'erreur.
                    st.code("".join(traceback.format_exception(type(erreur), erreur, erreur.__traceback__)))  # Affiche le détail.


# 9) ONGLET EXPLICATION          # Partie pédagogique pour le rapport et la soutenance.

with tab_explication:  # Début de l'onglet explication.
    st.subheader(" Explication du pipeline")  # Affiche le sous-titre.
    st.markdown(  # Affiche une explication pédagogique en Markdown.
        """
### 1. Chargement du dataset
Le programme cherche automatiquement des images ou vidéos dans le dossier indiqué.  
Les fichiers sont associés à une classe grâce aux noms de dossiers ou fichiers : `real`, `live`, `fake`, `spoof`, `attack`, etc.

### 2. Prétraitement
Chaque image est convertie en niveaux de gris, éventuellement recadrée sur le visage, redimensionnée, normalisée puis transformée en vecteur.

### 3. ACP / PCA
L'ACP réduit la dimension des images.  
Par exemple, une image de 96×96 pixels donne 9216 valeurs. Après ACP, on garde seulement les composantes principales qui conservent la majorité de l'information.

### 4. Classification SVM
Les caractéristiques extraites par l'ACP sont envoyées vers un SVM.  
Le SVM apprend une séparation entre les visages réels et les visages falsifiés.

### 5. Évaluation
L'interface affiche l'accuracy, la précision, le rappel, le F1-score, la matrice de confusion et la courbe de variance cumulée de l'ACP.
        """  # Fin du texte Markdown.
    )  # Fin de l'affichage Markdown.

    st.markdown("### Commande d'exécution")  # Affiche un titre pour la commande.
    st.code("streamlit run app_streamlit_acp.py", language="bash")  # Affiche la commande à lancer.

    st.markdown("### Bibliothèques à installer")  # Affiche un titre pour les dépendances.
    st.code(  # Affiche la commande pip.
        "pip install streamlit opencv-python scikit-learn pandas numpy matplotlib joblib",  # Commande d'installation.
        language="bash",  # Type du bloc de code.
    )  # Fin du bloc de code.