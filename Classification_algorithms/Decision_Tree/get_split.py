from Classification_algorithms.Decision_Tree.purity_measures import gini, mean_err_rate, entropy
import numpy as np
import pandas as pd


def get_split(df, criterion="infogain", nattrs=None):
    """Find best split on the given dataframe.

    Attributes:
        - df: the dataframe of smaples in the node to be split
        - criterion: spluis selection criterion
        - nattrs: flag to randomly limit the number of considered attributes. Used
          in random tree impementations.

    Returns:
        - If no split exists, return None.
        - If a split exists, return an instance of a subclass of AbstractSplit
    """
    # Implement termination criteria:
    # TermCrit1: Node is pure
    target_value_counts = df["target"].value_counts()
    if len(target_value_counts) == 1:
        return None
    # TermCrit2: No split is possible
    #    First get a list of attributes that can be split
    #    (i.e. attribute is not target and atribute can take more than one value)
    #
    #    The list of attributes on which we can split will also be handy for building random trees.
    possible_splits = [c for c in df.columns if
                       c not in ['target', 'weight'] and np.sum(pd.isna(df[c].unique()) == False) > 1]

    if nattrs is not None:
        np.random.shuffle(possible_splits)
        possible_splits = possible_splits[:nattrs]

    assert "target" not in possible_splits
    assert "weight" not in possible_splits

    #    Terminate early if none are possivle
    if not possible_splits:
        return None

    # Get the base purity measure and the purity function
    if criterion in ["infogain", "infogain_ratio"]:
        purity_fun = entropy
    elif criterion in ["mean_err_rate"]:
        purity_fun = mean_err_rate
    elif criterion in ["gini"]:
        purity_fun = gini
    else:
        raise Exception("Unknown criterion: " + criterion)
    base_purity = purity_fun(np.array([g_df['weight'].sum() for val, g_df in df.groupby('target')]))

    best_purity_gain = -1
    best_split = None

    # Random Forest support
    # restrict possible_splits to a few radomly selected attributes
    # if nattrs is not None:
    #     possible_splits = TODO

    for attr in possible_splits:
        if np.issubdtype(df[attr].dtype, np.number):
            split_sel_fun = get_numrical_split_and_purity
        else:
            split_sel_fun = get_categorical_split_and_purity

        split, purity_gain = split_sel_fun(
            df,
            base_purity,
            purity_fun,
            attr,
            normalize_by_split_entropy=criterion.endswith("ratio"),
        )

        if purity_gain > best_purity_gain:
            best_purity_gain = purity_gain
            best_split = split
    return best_split


def get_numrical_split_and_purity(
        df, parent_purity, purity_fun, attr, normalize_by_split_entropy=False
):
    """Find best split thereshold and compute the average purity after a split.
    Args:
        df: a dataframe
        parent_purity: purity of the parent node
        purity_fun: function to compute the purity
        attr: attribute over whihc to split the dataframe
        normalize_by_split_entropy: if True, divide the purity gain by the split
            entropy (to compute https://en.wikipedia.org/wiki/Information_gain_ratio)

    Returns:
        pair of (split, purity_gain)
    """
    from Classification_algorithms.Decision_Tree.numerical_split import NumericalSplit
    attr_df = df[[attr, "target", "weight"]].sort_values(attr)

    nans = attr_df.copy()[attr_df[attr].isna()]
    attr_df = attr_df.copy()[np.logical_not(attr_df[attr].isna())]
    attr_df = attr_df.set_index(np.arange(len(attr_df)))

    # Start with a split that puts all the samples into the right subtree
    right_counts = calculate_weights(df)
    left_counts = right_counts * 0

    nans_weights = pd.concat([left_counts, calculate_weights(nans)]).groupby(level=0).sum()

    best_split = None  # Will be None, or NumericalSplit(attr, best_threshold)
    best_purity_gain = -1

    w_1 = right_counts.sum()
    w_2 = 0
    total_weight = w_1 + w_2

    for row_i in range(len(attr_df) - 1):
        # Update the counts of targets in the left and right subtree and compute
        # the purity of the slipt for all possible thresholds!
        # Return the best split found.

        # Remember that the attribute may have duplicate values and all samples
        # with the same attribute value must end in the same subtree!
        row = attr_df.iloc[row_i]
        next_row = attr_df.iloc[row_i + 1]
        attribute_value = row[attr]
        next_attribute_value = next_row[attr]

        split_threshold = (attribute_value + next_attribute_value) / 2.0

        w_1 -= row['weight']
        w_2 += row['weight']

        # Consider the split at threshold, i.e. NumericalSplit(attr, split_threshold)

        # the loop should return the best possible split.

        # TODO: update left_counts and right_counts
        right_counts[row['target']] -= row['weight']
        left_counts[row['target']] += row['weight']

        # The split is possible if attribute_value != next_attribute_value
        if attribute_value != next_attribute_value:
            mean_child_purity = w_2 * purity_fun(left_counts + (w_2 / total_weight) * nans_weights)
            mean_child_purity += w_1 * purity_fun(right_counts + (w_1 / total_weight) * nans_weights)
            mean_child_purity /= total_weight
            purity_gain = parent_purity - mean_child_purity

            if purity_gain >= best_purity_gain:
                best_purity_gain = purity_gain
                best_split = NumericalSplit(attr, split_threshold)

        # TODO: now consider the split at split_threshold and save it if it the best one

    # if normalize_by_split_entropy:
    #     best_purity_gain /= entropy(df[attr].value_counts())

    return best_split, best_purity_gain


def get_categorical_split_and_purity(
        df, parent_purity, purity_fun, attr, normalize_by_split_entropy=False
):
    """Return a multivariate split and its purity.
    Args:
        df: a dataframe
        parent_purity: purity of the parent node
        purity_fun: function to compute the purity
        attr: attribute over whihc to split the dataframe
        normalize_by_split_entropy: if True, divide the purity gain by the split
            entropy (to compute https://en.wikipedia.org/wiki/Information_gain_ratio)

    Returns:
        pair of (split, purity_gain)
    """
    from Classification_algorithms.Decision_Tree.categorical_split import CategoricalMultivalueSplit
    split = CategoricalMultivalueSplit(attr)
    # Compute the purity after the split

    purity = []
    nans = df.copy()[df[attr].isna()]

    for value, group_df in df.groupby(attr):
        if nans.empty:
            w = group_df['weight'].sum() / (df['weight'].sum() - nans['weight'].sum())
            nans['weight'] = nans['weight'].apply(lambda x: x * w)
            group_df = pd.concat([group_df, nans])
            nans['weight'] = nans['weight'].apply(lambda x: x / w)
        purity.append(
            purity_fun(np.array([g_df['weight'].sum() for val, g_df in group_df.groupby('target')])) * group_df[
                'weight'].sum())

    mean_child_purity = np.sum(purity) / df['weight'].sum()

    # Note: when purity is measured by entropy, this corresponds to Mutual Information
    purity_gain = parent_purity - mean_child_purity
    if normalize_by_split_entropy:
        purity_gain /= entropy(np.array([g_df['weight'].sum() for val, g_df in df.groupby(attr)]))
    return split, purity_gain


def calculate_weights(df):
    return pd.Series(data={val: g_df['weight'].sum() for val, g_df in df.groupby('target')}, name='target')
