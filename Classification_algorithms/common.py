import pandas as pd


class AbstractClassifier:

    def __init__(self):
        self.train_x = None
        self.train_y = None

    def __str__(self):
        raise NotImplementedError

    def train(self, train_x: pd.DataFrame, train_y: pd.Series, **kwargs):
        """
            train_x: training dataset
            train_y: series of target values
        """
        self.train_x = train_x
        self.train_y = train_y

    def classify(self, test: pd.DataFrame, **kwargs):
        """
            test: test dataset
            module returns iterable object containing predictions
        """
        raise NotImplementedError
