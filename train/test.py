import joblib
import numpy as np

rf = joblib.load("datasets/processed/random_forest_model.joblib")
X_test = joblib.load("datasets/processed/X_test_path.joblib")

# Check unique predictions from EVERY individual tree
for i, tree in enumerate(rf.trees[:20]):  # first 20 trees
    preds = tree.predict(X_test.to_numpy()[:1000])
    print(f"Tree {i}: {np.unique(preds)}")