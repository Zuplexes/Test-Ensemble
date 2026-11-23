"""Decision Tree (Gini) ใช้เป็น base learner สำหรับ Bagging และ Boosting"""
import numpy as np


class SimpleTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def fit(self, X, y, K=None, sample_weight=None):
        self.K = int(y.max()) + 1 if K is None else K
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=float) / len(y)
        self.tree = self._build(X, y, sample_weight, depth=0)
        return self

    def _gini(self, y, w):
        total = w.sum()
        if total <= 0:
            return 0.0
        probs = np.bincount(y, weights=w, minlength=self.K) / total
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X, y, w):
        best_gain, best_feat, best_thresh = -1.0, None, None
        parent_gini = self._gini(y, w)
        total_w = w.sum()

        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            if len(thresholds) <= 1:
                continue
            midpoints = (thresholds[:-1] + thresholds[1:]) / 2.0
            for thresh in midpoints:
                left = X[:, feat] <= thresh
                right = ~left
                w_l, w_r = w[left].sum(), w[right].sum()
                if w_l == 0 or w_r == 0:
                    continue
                gain = parent_gini - (w_l / total_w) * self._gini(y[left], w[left]) - (w_r / total_w) * self._gini(y[right], w[right])
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        return best_feat, best_thresh

    def _build(self, X, y, w, depth):
        counts = np.bincount(y, weights=w, minlength=self.K)
        leaf_pred = int(np.argmax(counts))

        if depth >= self.max_depth or len(np.unique(y)) <= 1:
            return {"leaf": True, "pred": leaf_pred}

        feat, thresh = self._best_split(X, y, w)
        if feat is None:
            return {"leaf": True, "pred": leaf_pred}

        left = X[:, feat] <= thresh
        return {
            "leaf": False,
            "feat": feat,
            "thresh": thresh,
            "left": self._build(X[left], y[left], w[left], depth + 1),
            "right": self._build(X[~left], y[~left], w[~left], depth + 1),
        }

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["pred"]
        branch = node["left"] if x[node["feat"]] <= node["thresh"] else node["right"]
        return self._predict_one(x, branch)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X], dtype=int)