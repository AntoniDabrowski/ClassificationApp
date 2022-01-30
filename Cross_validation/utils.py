import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import datetime

from Classification_algorithms.common import AbstractClassifier


def cross_validation(x: pd.DataFrame, y: pd.DataFrame, root, pb, classifier: AbstractClassifier, folds, task_name, txt,
                     **kwargs):
    kf = KFold(n_splits=folds)
    err_rate_list = []

    total = kwargs['total']

    for train_index, test_index in kf.split(x):
        pb['value'] += 100 / (total * folds)
        txt['text'] = task_name + ' ' + str(np.round(pb['value'])) + '%'
        root.update_idletasks()
        classifier_instance = classifier()
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        classifier_instance.train(x_train, y_train, **kwargs)
        results = classifier_instance.classify(x_test, **kwargs)

        err_rate = 1 - np.sum(results == y_test) / results.size
        err_rate_list.append(err_rate)

    kwargs['err_rate'] = np.mean(err_rate_list).round(3)
    kwargs['standard_deviation'] = np.std(np.array(err_rate_list))
    kwargs['classifier'] = str(classifier)

    return classifier_instance, kwargs


def save_results(classifier: AbstractClassifier, test, path, **kwargs):
    now = datetime.datetime.now()
    filename = str(classifier) + '_' + now.strftime("%Y-%m-%d_%H-%M") + '.csv'

    results = classifier.classify(test, **kwargs)

    df = pd.DataFrame({'target': results})

    df.to_csv(path + '\\' + filename.replace(' ', '_'), index=False)
