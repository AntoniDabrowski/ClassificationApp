from Classification_algorithms.Decision_Tree.purity_measures import entropy, gini
from Classification_algorithms.Decision_Tree.get_split import get_split, calculate_weights
from Classification_algorithms.common import AbstractClassifier
import numpy as np
import graphviz
import pandas as pd


class DecisionTree(AbstractClassifier):
    """
    Auxiliary class that creates a Tree instance and performs on it training and classification
    in a similar way to other classifiers.
    """

    def __init__(self):
        super().__init__()
        self.tree = None

    def __str__(self):
        return "Decision Tree"

    def train(self, train_x: pd.DataFrame, train_y: pd.Series, criterion='infogain_ratio', **kwargs):
        train = train_x.copy()
        train['target'] = train_y
        tree = Tree(train, criterion=criterion)
        if kwargs.get('prune', True):
            tree.prune_with_confidence_interval()
        self.tree = tree

    def classify(self, test: pd.DataFrame, **kwargs):
        return np.array([self.tree.classify(test.iloc[i]) for i in range(len(test))])


class Tree:
    """
    Class that represents a single decision tree
    """

    def __init__(self, df, root=True, **kwargs):
        super().__init__()

        # Data can not have missing target values
        assert not df['target'].isnull().values.any()

        """
            initially all data points are equally weighted however when missing 
            value occurs the data point will be distributed to subnodes with
            decreased weight (proportional to size of node)
        """
        if root:
            # Weights initialization
            df = df.copy()
            df['weight'] = np.ones(len(df))

        # Collecting data for visualization
        if "all_targets" not in kwargs:
            kwargs["all_targets"] = sorted(df["target"].unique())

        # Save keyword arguments to build subtrees
        kwargs_orig = dict(kwargs)

        # Get kwargs we know about, remaining ones will be used for splitting
        self.all_targets = kwargs.pop("all_targets")

        # Save debug info for visualization
        self.weights = calculate_weights(df)
        self.counts = df["target"].value_counts()
        self.info = {
            "num_samples": len(df),
            "entropy": entropy(self.weights),
            "gini": gini(self.weights),
        }

        # if Tree contain data-samples of only one type there is no need for further splitting and following method
        # will return None. Otherwise it will return criterion that will partition data-samples into subtrees, hope-
        # fully purer than parent.
        self.split = get_split(df, **kwargs)

        if self.split:
            self.split.build_subtrees(df, kwargs_orig)

    def get_target_distribution(self, sample):
        # Case: leaf
        if self.split is None:
            return self.weights

        subtree = self.split(sample)

        # Case: inner node and sample got value for current attribute
        if subtree is not None:
            return subtree.get_target_distribution(sample)

        # Case: inner node and sample doesn't got value for current attribute
        sub_dfs = []
        for tree in self.split.iter_subtrees():
            sub_dfs.append(tree.get_target_distribution(sample))

        return pd.concat(sub_dfs).groupby(level=0).sum()

    def classify(self, sample):
        weights = self.get_target_distribution(sample)
        return weights.idxmax()

    def draw(self, depth=3, print_info=True):
        dot = graphviz.Digraph()
        self.add_to_graphviz(dot, print_info, depth)
        return dot

    def add_to_graphviz(self, dot, print_info, depth):
        freqs = self.counts / self.counts.sum()
        freqs = dict(freqs)
        colors = []
        freqs_info = []
        for i, c in enumerate(self.all_targets):
            freq = freqs.get(c, 0.0)
            if freq > 0:
                colors.append(f"{i % 9 + 1};{freq}")
                freqs_info.append(f"{c}:{freq:.2f}")
        colors = ":".join(colors)
        labels = [" ".join(freqs_info)]
        if print_info:
            for k, v in self.info.items():
                labels.append(f"{k} = {v}")
        if self.split:
            labels.append(f"split by: {self.split.attr}")
        dot.node(
            f"{id(self)}",
            label="\n".join(labels),
            shape="box",
            style="striped",
            fillcolor=colors,
            colorscheme="set19",
        )
        if self.split and depth > 1:
            self.split.add_to_graphviz(dot, self, print_info, depth - 1)

    def upper_confidenct_interval(self, f, N, z=0.5):
        return (f + ((z ** 2) / (2 * N)) \
                + z * ((f / N - (f ** 2) / N + z ** 2 / (4 * (N ** 2))) ** 0.5)) \
               / (1 + (z ** 2) / N)

    def prune_with_confidence_interval(self):
        """
            Prunning is a method that after creating a tree is checking whether it is not overfitted.
            Specifically it is checking if a leafs has higher upper confidence bound for a good pre-
            diction than its parents. If not, there is no need for that leaf. It is prunned and same
            process is considered on its father (new leaf)
        """
        if self.split:
            for w in self.split.iter_subtrees():
                w.prune_with_confidence_interval()

        N = np.sum(self.weights)
        parent_error = self.weights / np.sum(self.weights)
        parent_error = sorted(list(parent_error), reverse=True)
        parent_error = np.sum(parent_error[1:])
        parent_error = self.upper_confidenct_interval(parent_error, N)
        self.info['confidence_error'] = parent_error

        if self.split:
            children_error = 0
            for w in self.split.iter_subtrees():
                child_error = (np.sum(w.weights) / N) * w.info['confidence_error']
                children_error += child_error

            if children_error > parent_error:
                self.split = None
                self.info['splitted'] = True
