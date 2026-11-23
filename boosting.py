"""Boosting (SAMME): เทรนตามลำดับ -> เพิ่ม weight ตัวที่ทายผิด -> โหวตถ่วงน้ำหนักตาม alpha"""
import numpy as np
from tree import SimpleTree


class SequentialBoosting:
    def __init__(self, n_estimators=20, max_depth=3, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        self.K = int(y.max()) + 1
        N = len(X)
        w = np.ones(N, dtype=float) / N
        self.models = []
        self.alphas = []

        for _ in range(self.n_estimators):
            tree = SimpleTree(self.max_depth).fit(X, y, self.K, sample_weight=w)
            pred = tree.predict(X)
            incorrect = (pred != y).astype(float)
            err = np.dot(w, incorrect) / w.sum()

            # หยุดหากทายถูกทั้งหมด หรือแย่กว่าการสุ่ม (1 - 1/K)
            if err <= 1e-10:
                self.models.append(tree)
                self.alphas.append(1.0)
                break
            if err >= 1.0 - (1.0 / self.K):
                break

            # สูตร SAMME: alpha = ln((1 - err) / err) + ln(K - 1)
            alpha = np.log((1.0 - err) / err) + np.log(self.K - 1)
            w *= np.exp(alpha * incorrect)
            w /= w.sum()

            self.models.append(tree)
            self.alphas.append(alpha)

        return self

    def predict(self, X):
        scores = np.zeros((len(X), self.K), dtype=float)
        rows = np.arange(len(X))
        for tree, alpha in zip(self.models, self.alphas):
            scores[rows, tree.predict(X)] += alpha
        return scores.argmax(axis=1)