import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
import scipy.stats as sstats
from Classification_algorithms.Decision_Tree.decision_tree import Tree


class RandomForest:
    def __init__(self):
        self.forest = []

    def __str__(self):
        return "Random Forest"

    def train(self, train_x: pd.DataFrame, train_y: pd.Series, **kwargs):
        train = train_x.copy()
        train['target'] = train_y
        if kwargs.get('verbose',False):
            iterate_trees = tqdm(range(kwargs.get('trees_no', 30)), position=0,leave=False)
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
        tree = Tree(bootstrap, criterion=kwargs.get('criterion', 'infogain_ratio'), nattrs=kwargs.get('nattrs',1))
        return tree, out_of_bag

    def bootstrap_data(self, train: pd.DataFrame):
        N = len(train)
        train_idx = np.random.randint(0, N - 1, size=(N,))
        oob_idx = np.ones((N,), dtype=bool)
        oob_idx[train_idx] = 0
        oob_idx = np.where(oob_idx)[0]
        bootstrap = train.iloc[train_idx]
        out_of_bag = train.iloc[oob_idx]
        return bootstrap, out_of_bag

if __name__ == '__main__':
    train = pd.read_csv('../../data/train.csv')
    test = pd.read_csv('../../data/test.csv')

    train.rename(columns={"Survived": "target"}, inplace=True)
    train['Name'] = train['Name'].apply(lambda full_name: full_name.split(',')[0])
    test['Name'] = test['Name'].apply(lambda full_name: full_name.split(',')[0])

    train_x = train.iloc[:, 2:]
    train_y = train.iloc[:, 1]
    test = test.iloc[:, 1:]

    # train, test = train_test_split(train, test_size=0.3)

    RF = RandomForest()
    RF.train(train_x, train_y, trees_no=30, criterion='infogain_ratio', nattrs=1)

    results = RF.classify(test)

    df = pd.DataFrame({'PassengerId': np.arange(892, 1310), 'Survived': results})

    df.to_csv('..\Predictions\Random_Forest_2.csv', index=False)

    # print('nattr = 2')
    # RF.print_forest()
    # print(f'Forest OOB error rate: {RF.get_forest_OOB_err() * 100:.3f}%')
    # print(f'Mean trees error rate: {RF.mean_tree_errors()[0] * 100:.3f}%')
    # print(f'Mean trees agreement: {RF.forest_mean_agreement() * 100:.3f}%')
