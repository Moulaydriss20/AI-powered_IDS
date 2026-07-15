import numpy as np
import numpy.typing as npt
import joblib
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from src.models.isolation_forest import IsolationForest
from sklearn.metrics import classification_report #type: ignore
def compare_and_tune_threshold(X: npt.NDArray[np.float32], y_true: npt.NDArray[np.int16]):
    """
    Compare le modèle personnalisé avec Scikit-learn et ajuste automatiquement le seuil.
    y_true doit être encodé ainsi : -1 pour ATTACK, 1 pour BENIGN.
    """
    
    # ----------------------------------------------------
    # 1. CALCUL AUTOMATIQUE DU TAUX DE CONTAMINATION
    # ----------------------------------------------------
    # On calcule la proportion exacte d'attaques (-1) présentes dans l'échantillon réels
    n_samples = X.shape[0]
    n_attacks = np.sum(y_true == -1)
    auto_contamination = float(n_attacks / n_samples)
    
    print(f"--- Configuration du Seuil Automatique ---")
    print(f"Nombre total d'échantillons : {n_samples}")
    print(f"Nombre d'attaques réelles    : {n_attacks}")
    print(f"Taux de contamination calculé: {auto_contamination:.4f} (soit {auto_contamination*100:.2f}%)\n")

    # ----------------------------------------------------
    # 2. ENTRAÎNEMENT ET PRÉDICTION : MODÈLE PERSONNALISÉ
    # ----------------------------------------------------
    print("Ensemble personnalisé : Entraînement en cours...")
    # On instancie votre forêt avec la contamination automatique
    custom_forest = IsolationForest(n_trees=100, subsample_size=256, score_threshold=auto_contamination)
    custom_forest.fit(X)
    
    print("Ensemble personnalisé : Calcul des scores...")
    custom_scores = custom_forest.score(X)
    custom_preds = custom_forest.predict(X, custom_scores)

    # ----------------------------------------------------
    # 3. ENTRAÎNEMENT ET PRÉDICTION : SCIKIT-LEARN
    # ----------------------------------------------------
    print("Scikit-learn : Entraînement et calcul en cours...")
    # Configuration miroir pour une comparaison équitable
    # Note : Le score de sklearn est inversé par rapport au papier original, 
    # mais leur méthode predict() utilise exactement le même principe de percentile.
    sk_forest = SklearnIsolationForest(
        n_estimators=100, 
        max_samples=256, 
        contamination=auto_contamination, 
        random_state=42
    )
    sk_forest.fit(X)
    sk_preds = sk_forest.predict(X)

    # ----------------------------------------------------
    # 4. AFFICHAGE DES RAPPORTS DE CLASSIFICATION
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("RÉSULTATS : VOTRE MODÈLE PERSONNALISÉ (Seuil Auto)")
    print("="*50)
    print(classification_report(y_true, custom_preds, target_names=['ATTACK (-1)', 'BENIGN (1)']))#type: ignore
    print(f"Anomalies prédites par votre modèle : {np.sum(custom_preds == -1)}")

    print("\n" + "="*50)
    print("RÉSULTATS : SCIKIT-LEARN ISOLATION FOREST")
    print("="*50)
    print(classification_report(y_true, sk_preds, target_names=['ATTACK (-1)', 'BENIGN (1)']))#type: ignore
    print(f"Anomalies prédites par Scikit-learn : {np.sum(sk_preds == -1)}")
    
    return custom_scores, custom_preds, sk_preds

# Pour lancer l'expérience sur vos variables existantes :
# scores, preds, sk_preds = compare_and_tune_threshold(votre_matrice_X, vos_etiquettes_y)

def main():
    X_train = joblib.load("datasets/processed/X_train_path.joblib")
    y_train = joblib.load("datasets/processed/y_train_path.joblib")
    scaler = joblib.load("datasets/processed/scaler.joblib")

    X_train_scaled = scaler.transform(X_train).to_numpy()

    y_train = y_train.to_numpy()

    y_clean = np.ones(y_train.shape, dtype=np.int16)
    
    # Toutes les lignes où la valeur est comprise entre 1 et 5 deviennent des attaques (-1)
    attack_mask = (y_train >= 1) & (y_train <= 5)
    y_clean[attack_mask] = -1

    compare_and_tune_threshold(X_train_scaled, y_clean)

if __name__ == "__main__":
    #main()

    # Load model and Wednesday data
    rf = joblib.load('datasets/processed/random_forest_model.joblib')
    X_train = joblib.load('datasets/processed/X_train_path.joblib')
    y_train = joblib.load('datasets/processed/y_train_path.joblib')
    scaler = joblib.load('datasets/processed/scaler.joblib')

    X_train_scaled = scaler.transform(X_train).to_numpy()
    y_train_np = y_train.to_numpy()

    # Pick 100 known-benign and 100 known-attack samples
    benign_idx = np.where(y_train_np == 0)[0][:100]
    attack_idx = np.where(y_train_np != 0)[0][:100]

    # Get probabilities
    proba_benign = rf.predict_proba(X_train_scaled[benign_idx])
    proba_attack = rf.predict_proba(X_train_scaled[attack_idx])

    # Report P(attack) for each
    p_attack_on_benign = 1 - proba_benign[:, 0]
    p_attack_on_attack = 1 - proba_attack[:, 0]

    print(f"Benign samples — P(attack): mean={p_attack_on_benign.mean():.4f}, max={p_attack_on_benign.max():.4f}")
    print(f"Attack samples — P(attack): mean={p_attack_on_attack.mean():.4f}, min={p_attack_on_attack.min():.4f}")
    print(f"\nExpected: benign mean should be LOW (~0.0), attack mean should be HIGH (~1.0)")
