import numpy as np
import numpy.typing as npt
from src.models.decision_tree import DecisionTree
from typing import cast
from concurrent.futures import ProcessPoolExecutor, as_completed, Future

def _fit_single_tree(args: tuple[npt.NDArray[np.float32], npt.NDArray[np.int16], npt.NDArray[np.intp], int, int, int]) -> DecisionTree:
    X, y,idxs, max_depth, min_samples_split, n_features = args

    tree = DecisionTree(max_depth=max_depth,
                         min_sample_split=min_samples_split,
                         n_features=n_features
                        )
            
    tree.fit(X[idxs], y[idxs])
    return tree

class RandomForest:
    def __init__(self,
                 n_trees: int = 100,
                 max_depth: int = 10,
                 min_samples_split: int = 10,
                 n_classes: int = 0,
                 n_features: int | None = None
                 ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_classes = n_classes
        self.n_features = n_features
        self.trees: list[DecisionTree] = []

    def fit(self, X: npt.NDArray[np.float32], y: npt.NDArray[np.int16]) -> None:
        self.n_classes = int(y.max()) + 1
        n_features_per_tree = self._features_samples(X)
        n_samples = X.shape[0]
        complet = 0

        with ProcessPoolExecutor(max_workers=6) as executor:
            futures: set[Future[DecisionTree]]=set()
            for _ in range(self.n_trees):
                idxs = cast(npt.NDArray[np.intp], np.random.choice(n_samples, n_samples, replace= True))
                future: Future[DecisionTree] = executor.submit(
                    _fit_single_tree,
                    (X, y,idxs, self.max_depth, self.min_samples_split, n_features_per_tree)
                )
                futures.add(future)

            for future in as_completed(futures):
                tree = future.result()
                self.trees.append(tree)
                complet += 1

                print(f"Tree {complet}/{self.n_trees} trained", end="\r", flush=True)
    
    def _features_samples(self, X: npt.NDArray[np.float32]) -> int:
        if self.n_features is not None:
            return int(self.n_features)

        return int(np.sqrt(X.shape[1]))
    
    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.int16]:
        all_predictions = np.array([tree.predict(X) for tree in self.trees])

        n_samples = X.shape[0]

        final_predictions = np.zeros(n_samples, dtype=np.int16)

        for i in range(n_samples):

            sample_votes = all_predictions[:, i]

            final_predictions[i] = np.bincount(sample_votes).argmax()

            print(f"Prediction {i+1}/{n_samples} complete", end="\r")

        print()
        return final_predictions
    
    def predict_proba(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        all_predictions = np.array([tree.predict(X) for tree in self.trees])

        n_samples = X.shape[0]

        proba = np.zeros((n_samples, self.n_classes), dtype=np.float32)

        for i in range(n_samples):
            sample_vote = all_predictions[:, i]
            counts = np.bincount(sample_vote, minlength=self.n_classes)
            proba[i] = counts / self.n_trees

            if (i+1) % 10000 == 0:
                print(f"Proba {i+1}/{n_samples}", end="\r")

        print()
        
        return proba

    def get_feature_importances(self, n_features: int) -> npt.NDArray[np.float32]:
        importances = np.zeros(n_features, dtype=np.float32)

        if not self.trees:
            return importances
        
        for tree in self.trees:
            importances += tree.get_feature_importances(n_features)

        importances /= len(self.trees)

        sum_importances = np.sum(importances)
        if sum_importances > 0:
            importances /= sum_importances
 
        return importances


