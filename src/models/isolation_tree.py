import numpy as np
import numpy.typing as npt
from typing import cast

class Node:
    def __init__(self,
                 feature_idx : int | None,
                 threshold : float | None,
                 left : 'Node | Leaf | None',
                 right : 'Node | Leaf | None',
                 size : int 
                 ) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right   
        self.size = size  
    
class Leaf(Node):
    def __init__(self, size: int) -> None:
        super().__init__(feature_idx=None, threshold=None, left=None, right=None, size=size)

class IsolationTree:
    def __init__(self, max_depth: int = int(np.ceil(np.log2(256)))) -> None:
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: npt.NDArray[np.float32], depth: int=0) -> None:
        self.root = self._grow_tree(X, depth)
    
    def _grow_tree(self, X: npt.NDArray[np.float32], depth: int) -> Node | Leaf:
        n_samples, n_feature = X.shape

        if n_samples <= 1 or depth >= self.max_depth or np.all(X == X[0]):
            return Leaf(size=n_samples)
        

        feature_idx = cast(int, np.random.choice(n_feature, replace=False))
        feat_min, feat_max = np.min(X[:, feature_idx]), np.max(X[:, feature_idx])
        threshold = float(np.random.uniform(np.min(X[:, feature_idx]), np.max(X[:, feature_idx])))

        if feat_min == feat_max:
            return Leaf(size=n_samples)

        left_idx = np.where(X[:, feature_idx] <= threshold)[0]
        right_idx = np.where(X[:, feature_idx] > threshold)[0]

        left = self._grow_tree(X[left_idx], depth+1)
        right = self._grow_tree(X[right_idx], depth+1)

        return Node(feature_idx, threshold, left, right, size= n_samples)
    
    def path_length(self, X: npt.NDArray[np.float32], current_depth: int = 0, node: Node | Leaf | None = None) -> npt.NDArray[np.float32]:
        if node is None:
            lengths = np.full(X.shape[0], current_depth, dtype=np.float32)

            return lengths

        if isinstance(node, Leaf):
            lengths = np.full(X.shape[0], current_depth + self._correction(node.size), dtype=np.float32)

            return lengths
        
        if node.feature_idx is None or node.threshold is None:
            lengths = np.full(X.shape[0], current_depth, dtype=np.float32)

            return lengths
        
        lengths = np.zeros(X.shape[0], dtype=np.float32)
        
        left_mask = X[:, node.feature_idx] <= node.threshold
        right_mask = ~left_mask

        if np.any(left_mask):
            lengths[left_mask] = self.path_length(X[left_mask], current_depth+1, node=node.left)

        if np.any(right_mask):
            lengths[right_mask] = self.path_length(X[right_mask], current_depth+1, node=node.right)

        return lengths

    def _correction(self, size: int) -> float:
        if size <= 1:
            return 0.0
        
        if size == 2:
            return 1.0

        return 2 * (np.log(size - 1) + 0.5772156649) - (2 * (size - 1) / size)
    
