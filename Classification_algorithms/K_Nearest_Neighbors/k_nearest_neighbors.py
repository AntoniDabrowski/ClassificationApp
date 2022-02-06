import numpy as np
import scipy.stats as sstats
from scipy.spatial.distance import cdist
from Classification_algorithms.common import AbstractClassifier


class KNN(AbstractClassifier):

    def __init__(self):
        super().__init__()
        self.k = None
        self.distance = None

    def __str__(self):
        return "K-Nearest Neighbors"

    def train(self, train_x, train_y, k=[5], distance="euclidean", **kwargs):
        super().train(train_x, train_y)
        self.k = k if isinstance(k, list) else [k]
        self.distance = distance

    def create_proximity_matrix(self, test):
        proximity_matrix = cdist(self.train_x, test, metric=self.distance)
        return np.argsort(proximity_matrix, axis=0)

    def classify(self, x, **kwargs):
        distance_matrix = self.create_proximity_matrix(x)
        targets = np.array(self.train_y)[distance_matrix]

        results = []

        for k in self.k:
            predictions = sstats.mode(targets[:k, :]).mode
            predictions = predictions.ravel()
            results.append(predictions)

        return np.array(results[0])
