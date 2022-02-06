import itertools

import numpy as np
import pandas as pd


def get_data(train_x, train_y, test, encoder=None, cols=None, drop_invariant=True) -> (pd.DataFrame, pd.DataFrame):
    # Preparation of data for KNN approach
    x_list = [train_x, test]
    y_list = [train_y, None]

    if encoder is not None:
        x_merged = pd.concat(x_list)
        splits = np.cumsum([0] + [x.shape[0] for x in x_list])

        enc = encoder(cols=cols, drop_invariant=drop_invariant, return_df=True)
        enc.fit(x_merged)
        x_merged_transformed = enc.transform(x_merged)
        x_list = [x_merged_transformed.iloc[splits[i]:splits[i + 1]] for i in range(len(splits) - 1)]

    return list(itertools.chain(*[[x, y] for x, y in zip(x_list, y_list)]))


def group(series, dict_with_labels={}):
    l = []
    counter = 0
    for el in series:
        if el not in dict_with_labels:
            dict_with_labels[el] = counter
            counter += 1
        l.append(dict_with_labels[el])
    return pd.Series(l, name=series.name), dict_with_labels
