import os
import numpy as np
import numpy.typing as npt
import joblib
from src.models.isolation_forest import IsolationForest
from src.preprocessing.cleaner import cleaner
from src.preprocessing.encoder import Label_encoding
from src.preprocessing.spliter import spliter
from evaluate.metrics2 import evaluation
from evaluate.metric3 import evaluation_curve
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_isolation_results(X: npt.NDArray[np.float32], y_true: npt.NDArray[np.int16], y_pred: npt.NDArray[np.int16], scores: npt.NDArray[np.float32]) -> None:
    print("\n[INFO] Génération du graphique de visualisation...")
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    np.random.seed(42)
    plot_idx = np.random.choice(X_2d.shape[0], size=min(15000, X_2d.shape[0]), replace=False)
    
    X_plot = X_2d[plot_idx]
    y_true_plot = y_true[plot_idx]
    y_pred_plot = y_pred[plot_idx]
    scores_plot = scores[plot_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7)) #type: ignore

    benign_correct = (y_true_plot == 1) & (y_pred_plot == 1)
    attack_detected = (y_true_plot == -1) & (y_pred_plot == -1)
    errors = (y_true_plot != y_pred_plot)

    ax1.scatter(X_plot[benign_correct, 0], X_plot[benign_correct, 1], c='lightgray', label='Bénin (Correct)', alpha=0.5, s=15)
    ax1.scatter(X_plot[attack_detected, 0], X_plot[attack_detected, 1], c='crimson', label='Attaque Détectée (Vrai Positif)', alpha=0.7, s=15)
    ax1.scatter(X_plot[errors, 0], X_plot[errors, 1], c='darkorange', label='Erreur de Classification', alpha=0.8, s=25, marker='x')
    
    ax1.set_title("Analyse des Prédictions de l'IDS Personnalisé", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Composante Principale 1")
    ax1.set_ylabel("Composante Principale 2")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)


    sc = ax2.scatter(X_plot[:, 0], X_plot[:, 1], c=scores_plot, cmap='coolwarm', s=15, alpha=0.6)
    cbar = plt.colorbar(sc, ax=ax2) #type: ignore
    cbar.set_label("Score d'anomalie de votre modèle (Proche de 1 = Haut risque)", rotation=270, labelpad=20) #type: ignore
    
    ax2.set_title("Carte Thermique des Scores d'Isolation", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Composante Principale 1")
    ax2.set_ylabel("Composante Principale 2")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show() #type: ignore


def main():
    print("Loading data...")
    X_train_path = 'datasets/processed/X_train_path.joblib'
    X_test_path = 'datasets/processed/X_test_path.joblib'
    y_train_path = 'datasets/processed/y_train_path.joblib'
    y_test_path = 'datasets/processed/y_test_path.joblib'
    rf_path = "datasets/processed/random_forest_model.joblib"
    scaler_path = 'datasets/processed/scaler.joblib'

    required_files = [X_train_path, X_test_path,y_train_path, y_test_path, scaler_path]

    if all(os.path.exists(f) for f in required_files):
        X_train = joblib.load(X_train_path)
        X_test = joblib.load(X_test_path)
        y_train = joblib.load(y_train_path)
        y_test = joblib.load(y_test_path)
        rf = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)

    else:
        X_train , X_test, y_train, y_test = cleaner('datasets/raw/Wednesday-workingHours.pcap_ISCX.csv').pipe(Label_encoding).pipe(spliter)
        scaler = joblib.load(scaler_path)
        rf = joblib.load(rf_path)
        
    benign_mask = (y_train.to_numpy() == 0)
    X_train_scaled = scaler.transform(X_train).to_numpy()
    X_train_benign = X_train_scaled[benign_mask]
    X_test_scaled = scaler.transform(X_test).to_numpy()

    forest = IsolationForest()
    forest_benign_only = IsolationForest()
    print("--- Start Training ---")
    print("Training the forest with all the attacks")
    forest.fit(X_train_scaled)

    print("Training the forest with binign only")
    forest_benign_only.fit(X_train_benign)

# The anomaly should have a shorter path than normal points
    print("--- Start score computation ---")
    print("Compute score for the forest with all the attacks")
    scores = forest.score(X_test_scaled)

    print("--- Start prediction ---")
    print("Predict the forest with all the attacks")
    predicts = forest.predict(X_test_scaled, scores)
    print("--- Prediction ended ---")

    print(f"scores : \n{scores}")

    print(f"Predictions : \n{predicts}")

    y_samples = y_test.to_numpy()

    benign_scores = scores[y_samples == 0]
    attack_scores = scores[y_samples != 0]

    print(f"Average BENIGN score: {benign_scores.mean():.4f}")
    print(f"Average ATTACK score: {attack_scores.mean():.4f}")

    print(f"Predicted anomalies: {np.sum(predicts == -1)}")
    print(f"Actual attacks in sample: {np.sum(y_samples != 0)}")

    y_true_binary = np.where(y_samples == 0, 1, -1)

    classification = evaluation(y_true_binary, predicts)

    print(f"Classification report :\n{classification}")

    precisions, recalls, thresholds = evaluation_curve(X_test_scaled, y_samples, rf, forest, None)

    for i in range(0, len(thresholds), len(thresholds)//20):
        print(f"Threshold: {thresholds[i]:.4f} | Precision: {precisions[i]:.4f} | recall: {recalls[i]:.4f}")

    joblib.dump(forest, "datasets/processed/isolation_forest.joblib")
    joblib.dump(forest_benign_only, "datasets/processed/isolation_forest_benign_only.joblib")

    plot_isolation_results(X_test_scaled, y_samples, predicts, scores)

if __name__ == "__main__":
   main()