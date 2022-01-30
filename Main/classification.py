from Classification_algorithms.Decision_Tree.decision_tree import DecisionTree
from Classification_algorithms.Random_Forest.random_forest import RandomForest
from Classification_algorithms.Naive_Bayes_Classifier.naive_bayes import NaiveBayesClassifier
from Classification_algorithms.K_Nearest_Neighbors.k_nearest_neighbors import KNN
from Cross_validation.utils import cross_validation, save_results
from utils.data_preprocessing import get_data
from category_encoders import BackwardDifferenceEncoder, BaseNEncoder, BinaryEncoder, CountEncoder, HashingEncoder, \
    HelmertEncoder, OneHotEncoder, OrdinalEncoder, SumEncoder, PolynomialEncoder

import pandas as pd
import numpy as np
import itertools
import warnings


def classification(train_path, test_path, pred_path, dict_algorithms, root, pb, txt, mode):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train_x = train.loc[:, train.columns != 'target']
    train_y = train['target']
    list_of_algorithms = select_algorithms(dict_algorithms)
    errs = []
    upper_errs = []
    models = []
    test_encodings = []
    for algorithm, name in list_of_algorithms:
        print(name)
        pb['value'] = 0
        txt['text'] = name + ' ' + str(np.round(pb['value'])) + '%'
        root.update_idletasks()
        mean_error, upper_confidence_bound, model, test_encoding = algorithm(train_x, train_y, test, root, pb,
                                                                             name, txt, mode)
        errs.append(mean_error)
        upper_errs.append(upper_confidence_bound)
        models.append(model)
        test_encodings.append(test_encoding)
    best = np.argmin(upper_errs)
    best_model = models[best]
    print(str(best_model))
    print(errs[best])
    print(upper_errs[best])
    save_results(best_model, test_encodings[best], pred_path)


def knn(train_x, train_y, test, root, pb, task_name, txt, mode):
    encoders = {
        "BackwardDifferenceEncoder": BackwardDifferenceEncoder,
        "BaseNEncoder": BaseNEncoder,
        "BinaryEncoder": BinaryEncoder,
        "CountEncoder": CountEncoder,
        "HashingEncoder": HashingEncoder,
        "HelmertEncoder": HelmertEncoder,
        "OneHotEncoder": OneHotEncoder,
        "OrdinalEncoder": OrdinalEncoder,
        "SumEncoder": SumEncoder,
        "PolynomialEncoder": PolynomialEncoder
    }

    distances = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "euclidean",
                 "jensenshannon", "matching", "minkowski", "russellrao", "seuclidean", "sqeuclidean"]

    folds = 10
    if not mode:
        chosen = np.random.choice(list(encoders.keys()),3,replace=False)
        encoders = {key:encoders[key] for key in chosen}
        distances = np.random.choice(distances,3,replace=False)
        folds = 5


    upper_confidence_error = 1
    mean_error_rate = 1
    best_classifier = None
    winning_encoding = None

    total = len(encoders) * len(distances)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for encoder, distance in list(itertools.product(encoders.items(), distances)):
            try:
                key, encoder = encoder
                t_x, t_y, test_encoding, _ = get_data(train_x, train_y, test, encoder=encoder)
                classifier, result = cross_validation(t_x, t_y, root, pb, classifier=KNN, folds=folds, k=5,
                                                      encoder=encoder, distance=distance, total=total,
                                                      task_name=task_name, txt=txt)

                error_rate = result['err_rate']
                standard_deviation = result['standard_deviation']
                confidence_upper_bound = error_rate + 2 * standard_deviation
                if confidence_upper_bound < upper_confidence_error:
                    winning_encoding = test_encoding
                    upper_confidence_error = confidence_upper_bound
                    mean_error_rate = error_rate
                    best_classifier = classifier
            except RuntimeError:
                continue
    return mean_error_rate, upper_confidence_error, best_classifier, winning_encoding


def decision_tree(train_x, train_y, test, root, pb, task_name, txt, mode):
    # return mean_error, upper_confidence_bound
    k = 10

    upper_confidence_error = 1
    mean_error_rate = 1
    best_classifier = None

    criterions = ["infogain", "infogain_ratio", "mean_err_rate", "gini"]
    if not mode:
        criterions = ['infogain_ratio']
        k = 5

    total = len(criterions)
    for criterion in criterions:
        classifier, result = cross_validation(train_x, train_y, root, pb, classifier=DecisionTree, folds=k,
                                              criterion=criterion, total=total, task_name=task_name, txt=txt)
        error_rate = result['err_rate']
        standard_deviation = result['standard_deviation']
        confidence_upper_bound = error_rate + 2 * standard_deviation
        if confidence_upper_bound < upper_confidence_error:
            upper_confidence_error = confidence_upper_bound
            mean_error_rate = error_rate
            best_classifier = classifier
    return mean_error_rate, upper_confidence_error, best_classifier, test


def random_forest(train_x, train_y, test, root, pb, task_name, txt, mode):
    k = 10

    upper_confidence_error = 1
    mean_error_rate = 1
    best_classifier = None
    criterions = ["infogain", "infogain_ratio", "mean_err_rate", "gini"]
    if not mode:
        criterions = ['infogain_ratio']
        k = 5
    total = len(criterions)
    for criterion in criterions:
        classifier, result = cross_validation(train_x, train_y, root, pb, classifier=RandomForest, folds=k,
                                              criterion=criterion,
                                              nattrs=1, trees_no=25, verbose=False, total=total, task_name=task_name,
                                              txt=txt)
        error_rate = result['err_rate']
        standard_deviation = result['standard_deviation']
        confidence_upper_bound = error_rate + 2 * standard_deviation
        if confidence_upper_bound < upper_confidence_error:
            upper_confidence_error = confidence_upper_bound
            mean_error_rate = error_rate
            best_classifier = classifier
    return mean_error_rate, upper_confidence_error, best_classifier, test


def naive_bayes_classifier(train_x, train_y, test, root, pb, task_name, txt, mode):
    encoders = {
        "BackwardDifferenceEncoder": BackwardDifferenceEncoder,
        "BaseNEncoder": BaseNEncoder,
        "BinaryEncoder": BinaryEncoder,
        "CountEncoder": CountEncoder,
        "HashingEncoder": HashingEncoder,
        "HelmertEncoder": HelmertEncoder,
        "OneHotEncoder": OneHotEncoder,
        "OrdinalEncoder": OrdinalEncoder,
        "SumEncoder": SumEncoder,
        "PolynomialEncoder": PolynomialEncoder
    }

    folds = 10
    if not mode:
        chosen = np.random.choice(list(encoders.keys()),3,replace=False)
        encoders = {key:encoders[key] for key in chosen}
        folds = 5

    upper_confidence_error = 1
    mean_error_rate = 1
    best_classifier = None
    winning_encoding = None
    total = len(encoders)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for _, encoder in encoders.items():
            try:
                t_x, t_y, test_encoding, _ = get_data(train_x, train_y, test, encoder=encoder)
                classifier, result = cross_validation(t_x, t_y, root, pb, classifier=NaiveBayesClassifier, folds=folds,
                                                      total=total, task_name=task_name, txt=txt)
                error_rate = result['err_rate']
                standard_deviation = result['standard_deviation']
                confidence_upper_bound = error_rate + 2 * standard_deviation
                if confidence_upper_bound < upper_confidence_error:
                    winning_encoding = test_encoding
                    upper_confidence_error = confidence_upper_bound
                    mean_error_rate = error_rate
                    best_classifier = classifier
            except RuntimeError:
                continue
    return mean_error_rate, upper_confidence_error, best_classifier, winning_encoding


def select_algorithms(dict_algorithms):
    list_of_algorithms = []
    if dict_algorithms['K-nearest neighbors']:
        list_of_algorithms.append((knn,'K-nearest neighbors'))
    if dict_algorithms['Naive Bayes Classifier']:
        list_of_algorithms.append((naive_bayes_classifier,'Naive Bayes Classifier'))
    if dict_algorithms['Decision Tree']:
        list_of_algorithms.append((decision_tree,'Decision Tree'))
    if dict_algorithms['Random Forest']:
        list_of_algorithms.append((random_forest,'Random Forest'))
    return list_of_algorithms


if __name__ == '__main__':
    train_path = r"C:/Users/user/Studia/Semestr V/Python/Projekt/Code/Example_dataset/train.csv"
    test_path = r"C:/Users/user/Studia/Semestr V/Python/Projekt/Code/Example_dataset/test.csv"
    pred_path = r"C:/Users/user/Studia/Semestr V/Python/Projekt/Code/Example_dataset"
    dict_algorithms = {'K-nearest neighbors': 0, 'Naive Bayes Classifier': 1, 'Decision Tree': 0, 'Random Forest': 0}

    classification(train_path, test_path, pred_path, dict_algorithms)
