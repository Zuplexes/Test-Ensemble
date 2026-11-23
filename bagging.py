"""Bagging: bootstrap -> เทรนอิสระต่อกัน -> โหวตเสียงข้างมาก
ถ้า random_state เท่ากัน Sequential และ Parallel ได้โมเดลเหมือนกัน (ต่างกันแค่ความเร็ว)"""
import os
import concurrent.futures
import numpy as np
from tree import SimpleTree

_DATA = {}


def _fit_one(X, y, depth, K, seed):
    idx = np.random.RandomState(seed).randint(0, len(X), len(X))   # with replacement
    return SimpleTree(depth).fit(X[idx], y[idx], K)


def _init_worker(*args):
    _DATA["args"] = args            # ส่งข้อมูลให้ worker ครั้งเดียว


def _worker(seed):
    return _fit_one(*_DATA["args"], seed)


class SequentialBagging:
    def __init__(self, n_estimators=20, max_depth=3, random_state=0):
        self.n_estimators, self.max_depth, self.random_state = n_estimators, max_depth, random_state

    def _train(self, X, y, seeds):
        return [_fit_one(X, y, self.max_depth, self.K, s) for s in seeds]

    def fit(self, X, y):
        self.K = int(y.max()) + 1
        seeds = np.random.RandomState(self.random_state).randint(0, 2**31 - 1, self.n_estimators)
        self.models = self._train(X, y, seeds)
        return self

    def predict(self, X):
        votes = np.zeros((len(X), self.K), dtype=int)
        rows = np.arange(len(X))
        for m in self.models:
            votes[rows, m.predict(X)] += 1
        return votes.argmax(axis=1)


class ParallelBagging(SequentialBagging):
    def _train(self, X, y, seeds):
        with concurrent.futures.ProcessPoolExecutor(
            os.cpu_count(), initializer=_init_worker,
            initargs=(X, y, self.max_depth, self.K),
        ) as ex:
            return list(ex.map(_worker, seeds))