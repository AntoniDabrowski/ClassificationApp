from Classification_algorithms.Decision_Tree.abstract_split import AbstractSplit
from Classification_algorithms.Decision_Tree.decision_tree import Tree
import pandas as pd


class CategoricalMultivalueSplit(AbstractSplit):
    def build_subtrees(self, df, subtree_kwargs):
        self.subtrees = {}
        nans = df.copy()[df[self.attr].isna()]
        for group_name, group_df in df.groupby(self.attr):
            if not pd.isna(group_name):
                if nans.empty:
                    w = group_df['weight'].sum() / (df['weight'].sum() - nans['weight'].sum())
                    nans['weight'] = nans['weight'].apply(lambda x: x * w)
                    group_df = pd.concat([group_df, nans])
                    nans['weight'] = nans['weight'].apply(lambda x: x / w)
                child = Tree(group_df, root=False, **subtree_kwargs)
                self.subtrees[group_name] = child

    def __call__(self, x):
        # Return the subtree for the given example
        val = x[self.attr]
        if pd.isna(val):
            return None
        return self.subtrees.get(val, None)

    def iter_subtrees(self):
        return self.subtrees.values()

    def add_to_graphviz(self, dot, parent, print_info, depth):
        for split_name, child in self.subtrees.items():
            child.add_to_graphviz(dot, print_info, depth)
            dot.edge(f"{id(parent)}", f"{id(child)}", label=f"{split_name}")
