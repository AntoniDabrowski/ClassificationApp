from Classification_algorithms.Decision_Tree.decision_tree import Tree
from Classification_algorithms.Decision_Tree.abstract_split import AbstractSplit
import pandas as pd


class NumericalSplit(AbstractSplit):
    def __init__(self, attr, th):
        super(NumericalSplit, self).__init__(attr)
        self.th = th

    def build_subtrees(self, df, subtree_kwargs):
        nans = df.copy()[df[self.attr].isna()]
        lower_df = df[df[self.attr] <= self.th]
        upper_df = df[df[self.attr] > self.th]

        if not nans.empty:
            w_l = lower_df['weight'].sum() / (lower_df['weight'].sum() + upper_df['weight'].sum())
            w_u = 1 - w_l
            nans['weight'] = nans['weight'].apply(lambda x: x * w_l)
            lower_df = pd.concat([lower_df, nans])
            nans['weight'] = nans['weight'].apply(lambda x: x / w_l)

            nans['weight'] = nans['weight'].apply(lambda x: x * w_u)
            upper_df = pd.concat([upper_df, nans])
            nans['weight'] = nans['weight'].apply(lambda x: x / w_u)

        # # Different approach:
        # if df[self.attr].mean() <= self.th:
        #     lower_df = pd.concat([lower_df,nans])
        # else:
        #     upper_df = pd.concat([upper_df,nans])

        self.subtrees = (
            Tree(lower_df, root=False, **subtree_kwargs),
            Tree(upper_df, root=False, **subtree_kwargs),
        )

    def __call__(self, x):
        # return the subtree for the data sample `x`
        if pd.isna(self.attr):
            return None
        return self.subtrees[int(x[self.attr] > self.th)]

    def __str__(self):
        return f"NumericalSplit: {self.attr} <= {self.th}"

    def iter_subtrees(self):
        return self.subtrees

    def add_to_graphviz(self, dot, parent, print_info, depth):
        self.subtrees[0].add_to_graphviz(dot, print_info, depth)
        dot.edge(f"{id(parent)}", f"{id(self.subtrees[0])}", label=f"<= {self.th:.2f}")
        self.subtrees[1].add_to_graphviz(dot, print_info, depth)
        dot.edge(f"{id(parent)}", f"{id(self.subtrees[1])}", label=f"> {self.th:.2f}")
