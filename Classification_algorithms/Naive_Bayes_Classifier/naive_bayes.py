import pandas as pd
import numpy as np
from Classification_algorithms.common import AbstractClassifier


class NaiveBayesClassifier(AbstractClassifier):
    def __init__(self):
        super().__init__()
        self.probabilities = {}
        self.prob_of_survival = None

    def __str__(self):
        return "Naive Bayes Classifier"

    def train(self, train_x: pd.DataFrame, train_y: pd.Series, threshold=1, **kwargs):
        super().train(train_x, train_y)

        self.prob_of_survival = train_y.mean()

        self.probabilities = {}
        for name in train_x.columns:
            d = {key: [] for key in train_x[name].unique()}
            for el, tar in zip(train_x[name], train_y):
                if not pd.isna(el):
                    d[el].append(tar)
            self.probabilities[name] = {key: np.mean(d[key]) for key in d.keys() if len(d[key]) >= threshold}

    def classify(self, test: pd.DataFrame, **kwargs):
        predictions = []
        for _, row in test.iterrows():
            naive_part_survived = [np.log(self.probabilities[row_name].get(row_value, 0.5) + 1e-100) for
                                   row_name, row_value in
                                   zip(row.index[1:], row[1:])]
            naive_part_died = [np.log((1 - self.probabilities[row_name].get(row_value, 0.5)) + 1e-100) for
                               row_name, row_value in
                               zip(row.index[1:], row[1:])]

            prob_of_observation_cond_survived = np.exp(np.sum(naive_part_survived))
            prob_of_observation_cond_died = np.exp(np.sum(naive_part_died))

            probability_of_observation = self.prob_of_survival * prob_of_observation_cond_survived + (
                    1 - self.prob_of_survival) * prob_of_observation_cond_died

            assumption = (prob_of_observation_cond_survived * self.prob_of_survival) / probability_of_observation

            predictions.append(int(assumption > 0.5))

        return np.array(predictions)
