import numpy as np
import numpy.typing as npt
from typing import cast
from src.models.isolation_tree import IsolationTree

class IsolationForest:
    def __init__(self, n_trees: int = 100, subsample_size: int = 256, contamination: float = 0.36) -> None:
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.contamination = contamination
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

        print("---Training ended\n")

    def score(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        n_samples, _ = X.shape

        avg_path_length = np.zeros(n_samples, dtype=np.float32)

        for i, row in enumerate(X):
            print(f"Computing score {i+1}/{n_samples}", end="\r")
            avg_path_length[i] = np.mean([tree.path_length(row, node=tree.root) for tree in self.trees])

        
        scores = 2 ** (- avg_path_length / self._correction(self.subsample_size))

        print(f"---Score computation ended\n")
        return scores

    def _correction(self, size: int) -> float:
        if size == 1:
            return 1
        
        return 2 * (np.log(size - 1) + 0.5772156649) - (2 * (size - 1) / size)
    
    def predict(self, X: npt.NDArray[np.float32], scores : npt.NDArray[np.float32]) -> npt.NDArray[np.int16]:
        threshold = np.percentile(scores, 100 * (1 - self.contamination))

        predicts = np.where(scores >= threshold, -1, 1)
        
        return predicts