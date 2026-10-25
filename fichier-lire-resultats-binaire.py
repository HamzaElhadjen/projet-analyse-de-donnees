import joblib
import numpy as np

# Chemin corrigé avec slash (ou raw string)
fichier = "C:/Users/hamza/OneDrive/Images/modele_acp_anti_spoofing_streamlit (1).joblib"

try:
    contenu = joblib.load(fichier)
    print("Type du contenu :", type(contenu))
    
    if isinstance(contenu, dict):
        print("\nClés dans le dictionnaire :", list(contenu.keys()))
        for k, v in contenu.items():
            if isinstance(v, (int, float, str, np.number)):
                print(f"{k} : {v}")
        # Si le modèle est présent
        modele = contenu.get("modele")
        if modele is not None:
            print("\n=== Infos sur le modèle ===")
            # Récupérer l'ACP
            if hasattr(modele, "named_steps") and "acp" in modele.named_steps:
                acp = modele.named_steps["acp"]
                print("Nombre composantes ACP :", acp.n_components_)
                print("Variance PC1 :", acp.explained_variance_ratio_[0])
                print("Variance PC2 :", acp.explained_variance_ratio_[1] if acp.n_components_ > 1 else "N/A")
                print("Variance cumulée :", np.sum(acp.explained_variance_ratio_))
    else:
        print("Contenu :", contenu)
        
except Exception as e:
    print("Erreur :", e)