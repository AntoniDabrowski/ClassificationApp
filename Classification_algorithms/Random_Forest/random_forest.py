import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
import scipy.stats as sstats
from Classification_algorithms.Decision_Tree.decision_tree import Tree
from Classification_algorithms.common import AbstractClassifier


class RandomForest(AbstractClassifier):
    """
        Classification algorithm that creates many partially random decision trees
        that connected to one 'majority vote classifier' gives very accurate predictions.
        Advantage of this approach is that I can check confidence of a prediction based on
        number of trees that 'voted' for particular option.
    """

    def __init__(self):
        self.forest = []

    def __str__(self):
        return "Random Forest"

    def train(self, train_x: pd.DataFrame, train_y: pd.Series, **kwargs):
        train = train_x.copy()
        train['target'] = train_y
        if kwargs.get('verbose', False):
            iterate_trees = tqdm(range(kwargs.get('trees_no', 30)), position=0, leave=False)
        else:
            iterate_trees = range(kwargs.get('trees_no', 30))
        for _ in iterate_trees:
            tree, out_of_bag = self.create_tree(train, **kwargs)
            self.forest.append(tree)

    def classify(self, test: pd.DataFrame, **kwargs):
        res_targets = np.array([[tree.classify(test.iloc[i]) for i in range(len(test))] for tree in self.forest])

        results = []
        for i in range(len(test)):
            trees_predictions = res_targets[:, i]
            majority_vote = sstats.mode(trees_predictions)[0][0]
            results.append(majority_vote)

        return np.array(results)

    def create_tree(self, train: pd.DataFrame, **kwargs):
        bootstrap, out_of_bag = self.bootstrap_data(train)
        tree = Tree(bootstrap, criterion=kwargs.get('criterion', 'infogain_ratio'), nattrs=kwargs.get('nattrs', 1))
        return tree, out_of_bag

    def bootstrap_data(self, train: pd.DataFrame):
        # One way of providing diversity between trees is to make each based on partially different datasets
        # It is what bootstrap method provides.
        N = len(train)
        train_idx = np.random.randint(0, N - 1, size=(N,))
        oob_idx = np.ones((N,), dtype=bool)
        oob_idx[train_idx] = 0
        oob_idx = np.where(oob_idx)[0]
        bootstrap = train.iloc[train_idx]
        out_of_bag = train.iloc[oob_idx]
        return bootstrap, out_of_bag
