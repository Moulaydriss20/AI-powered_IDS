import numpy as np
import numpy.typing as npt
from typing import cast
from src.models.isolation_tree import IsolationTree

class IsolationForest:
    def __init__(self, n_trees: int = 100, subsample_size: int = 256, score_threshold: float = 0.447) -> None:
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.score_threshold = score_threshold
        self.trees: list[IsolationTree] = []

    def fit(self, X: npt.NDArray[np.float32]) -> None:
        n_samples = X.shape[0]
        self.subsample_size = min(self.subsample_size, n_samples)

        for i in range(self.n_trees):
            print(f"Training tree {i+1}/{self.n_trees}", end="\r")
            idxs = cast(npt.NDArray[np.int32], np.random.choice(n_samples, size=self.subsample_size, replace=False))
            tree = IsolationTree()
            
            tree.fit(X[idxs])

            self.trees.append(tree)
        print()
        
        print("---Training ended\n")

    def score(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        all_paths = np.array([tree.path_length(X, node=tree.root) for tree in self.trees])

        avg_path_length = np.mean(all_paths, axis=0)
        
        c_factor = self._correction(self.subsample_size)
        if c_factor == 0:
            scores = np.zeros(X.shape[0], dtype=np.float32)

            return scores
        
        scores = 2 ** (- avg_path_length / c_factor)

        print("-- Score computation ended ---")

        return scores

    def _correction(self, size: int) -> float:
        if size == 1:
            return 1
        
        return 2 * (np.log(size - 1) + 0.5772156649) - (2 * (size - 1) / size)
    
    def predict(self, X: npt.NDArray[np.float32], scores : npt.NDArray[np.float32]) -> npt.NDArray[np.int16]:

        predicts = np.where(scores >= self.score_threshold, -1, 1)
        
        return predicts