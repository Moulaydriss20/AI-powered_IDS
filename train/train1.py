import os
import time
from src.preprocessing.cleaner import cleaner
from src.preprocessing.encoder import Label_encoding
from src.preprocessing.spliter import spliter
from src.feature.scaler import feature_scaling
from src.models.random_forest import RandomForest
from evaluate.metrics import evaluation
from src.feature.selector import get_feature_importances
import joblib

def main():
    print("Loading data...")
    X_train_path = 'datasets/processed/X_train_path.joblib'
    X_test_path = 'datasets/processed/X_test_path.joblib'
    y_train_path = 'datasets/processed/y_train_path.joblib'
    y_test_path = 'datasets/processed/y_test_path.joblib'

    required_files = [X_train_path, X_test_path, y_train_path, y_test_path]

    if all(os.path.exists(f) for f in required_files):
        X_train = joblib.load(X_train_path)
        X_test = joblib.load(X_test_path)
        y_train = joblib.load(y_train_path)
        y_test = joblib.load(y_test_path)

    else:
        X_train , X_test, y_train, y_test = cleaner('datasets/raw/Wednesday-workingHours.pcap_ISCX.csv').pipe(Label_encoding).pipe(spliter)

    X_train_scaled, X_test_scaled = feature_scaling(X_train, X_test)

    rf = RandomForest()

    start = time.perf_counter()

    print(f"Training random forest ({rf.n_trees})")
    rf.fit(X_train_scaled.to_numpy(), y_train.to_numpy())

    end = time.perf_counter()

    timer = (end - start) / 3600

    predictions = rf.predict(X_test_scaled.to_numpy())

    accuracy, classification, confusion = evaluation(y_test.to_numpy(), predictions)

    importances = get_feature_importances(rf, X_train_scaled)

    print(f"The top 20 features: {importances[:20]}")

    print(f"Model Accuracy : {accuracy:.2f}")

    print(f"Model classification report : {classification}")

    print(f"Model confusion matrix : {confusion}")

    print(f"Training time : {timer} h")

    print("About to save model...")
    joblib.dump(rf, "datasets/processed/random_forest_model.joblib")
    print("Model saved successfully.")

if __name__ == '__main__':
    main()