import numpy as np
import numpy.typing as npt
from typing import Any, cast

#Create a node.
class Node:
    #Node characteristics.
    def __init__(self,
                 feature_index: int | None,
                 n_samples: int,
                 gain: float ,
                 threshold: float | None ,
                 left: 'Leaf  | Node | None',
                 right: 'Leaf | Node | None') -> None:
        self.feature_index = feature_index
        self.n_samples = n_samples
        self.gain = gain
        self.threshold = threshold
        self.left = left
        self.right = right

#Create a leaf node.
class Leaf(Node):
    #Leaf characteristics.
    def __init__(self,
                 label: int,
                 n_samples: int
                ) -> None:
        super().__init__(feature_index=None, n_samples=n_samples, gain=0, threshold=None, left=None, right=None)
        self.label = label

#Create a tree.
class DecisionTree:
    #Tree characteristics
    def __init__(self, max_depth: int = 10, min_sample_split: int = 10, n_features: int | None = None) -> None:
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_features = n_features
        self.root = None

    def fit(self, X: npt.NDArray[np.float32], y: npt.NDArray[np.int16]) -> None:
        self.n_features = cast(npt.NDArray[np.intp], np.arange(X.shape[1])) if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
    

    #Create left and right nodes.
    def _grow_tree(self, X: npt.NDArray[np.float32], y: npt.NDArray[np.int16], depth: int = 0, n_feature: int | None= None) -> Leaf | Node:
        n_labels, n_samples = len(np.unique(y)), X.shape[0]

        all_features = cast(npt.NDArray[np.int16], np.arange(X.shape[1]))

        if n_feature is None:
            self.n_features = all_features
        else:
            self.n_features = np.random.choice(all_features, size=n_feature, replace=False)


        if n_labels == 1:

            return Leaf(label=y[0], n_samples=n_samples)
        
        if depth >= self.max_depth:

            return Leaf(label=self._most_common_label(y), n_samples=n_samples)
        
        if n_samples < self.min_sample_split:

            return Leaf(label=self._most_common_label(y), n_samples=n_samples)
        
        feature_idx, threshold, gain = self.best_split(X, y, self.n_features)

        if gain == 0:
        
            return Leaf(label=self._most_common_label(y), n_samples=n_samples)

        left_indx, right_indxs = self._split(X[:, feature_idx], threshold)

        left = self._grow_tree(X[left_indx], y[left_indx], depth+1, n_feature)
        right = self._grow_tree(X[right_indxs], y[right_indxs],depth+1, n_feature)


        return Node(feature_idx ,n_samples, gain, threshold, left, right)

    #Calculate the impurity of a node.
    def gini_impurity(self, y:npt.NDArray[np.int16]) -> float:
        hist = np.bincount(y)

        ps = hist / len(y)

        return 1 - (np.sum([ p**2 for p in ps ]))
    
    #Calculate the best split possible.
    def best_split(self, X: npt.NDArray[np.float32], y: npt.NDArray[np.int16], features_indxs: npt.NDArray[np.int16]) -> tuple[int | None, float |None, float]:
        best_gain = 0
        best_feature, best_threshold = None, None

        
        for feature in features_indxs:
           #Get each column.
           X_column = X[:, feature]
           #get
           thresholds = np.percentile(X_column, np.linspace(10, 90, 10))

           for threshold in thresholds:

               gain = self._inforamtion_gain(X_column, y, threshold)

               if gain > best_gain :
                   best_gain = gain
                   best_feature = feature
                   best_threshold = threshold
                

        return best_feature, best_threshold, best_gain
    
    #Calculate the information gain.
    def _inforamtion_gain(self, X_column: npt.NDArray[np.float32], y: npt.NDArray[np.int16],threshold: float) -> float:
        #Parent gini.
        parent_gini = self.gini_impurity(y)

        #Get the split indexs.
        left_idxs, right_idxs = self._split(X_column, threshold)

        #Skip if one of the sides are empty.
        if len(left_idxs) == 0 or len(right_idxs) == 0 :

            return 0
        
        #Left and right gini.
        left_gini = self.gini_impurity(y[left_idxs])
        right_gini = self.gini_impurity(y[right_idxs])

        #Child gini.
        weighted_gini = (len(left_idxs)/len(y)) * left_gini + (len(right_idxs) / len(y)) * right_gini

        #Return the gain.
        return parent_gini - weighted_gini
    
    #Split the node based on a criteria.
    def _split(self, X_column: npt.NDArray[np.float32], threshold: Any) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
        #Get the left and right indexs based on a criteria.
        left_idxs = cast(npt.NDArray[np.int16], np.where(X_column <= threshold)[0]) #Left = features columns <= threshold.
        right_idxs = cast(npt.NDArray[np.int16], np.where(X_column > threshold)[0]) #right = features columns >= threshold.

        #Return the left and right indexs
        return left_idxs, right_idxs
    
    #Calculate teh most common label in the result array.
    def _most_common_label(self, y: npt.NDArray[np.int16]) -> int:
        #return the most common label
        return int(np.bincount(y).argmax())
    
    def _traverse_tree(self, X: npt.NDArray[np.float32] , node: Node | Leaf | None) -> int:
        if node is None:
            return 0

        if isinstance(node, Leaf):
            return node.label
        
        if X[cast(int, node.feature_index)] <= cast(float, node.threshold):
            return self._traverse_tree(X, node.left)
        
        return self._traverse_tree(X, node.right)
    
    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.int16]:
        if self.root is None:
            return cast(npt.NDArray[np.int16], np.zeros(X.shape[0], dtype=np.int16))

        return np.array([self._traverse_tree(x, self.root) for x in X], dtype=np.int16)
    
    def get_feature_importances(self, n_features : int) -> npt.NDArray[np.float32]:
        importances = np.zeros(n_features, dtype=np.float32)

        if not self.root:
            return importances
        
        def _traverse(node: Node | Leaf, imp: npt.NDArray[np.float32]):
            if isinstance(node, Leaf):
                return

            imp[node.feature_index] += node.gain * node.n_samples

            if node.left:
                _traverse(node.left, imp)
            if node.right:
                _traverse(node.right, imp)
        
        _traverse(self.root, importances)

        total_samples = self.root.n_samples

        if total_samples > 0:
            importances /= total_samples

        return importances