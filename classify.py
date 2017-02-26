from itertools import cycle

from time import time
from operator import itemgetter

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Perceptron

from sklearn.tree import DecisionTreeClassifier
#from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import  cross_val_score
from sklearn.datasets import load_digits

from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from scipy import interp
from sklearn import svm
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score, classification_report, confusion_matrix, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

import csv
import matplotlib.pyplot as plt
import numpy as np

# K-Fold Constants
concat_test_train = True
block_size_k = 1500

# File names
vocab_file = 'out_vocab_5.txt'
train_label_file = 'out_classes_5.txt'
train_feature_matrix_file = 'out_bag_of_words_5.csv'

test_true_label_file = 'test_classes_0.txt'
test_feature_matrix_file = 'test_bag_of_words_0.csv'


def _load_file_as_array(file_name):
    with open(file_name, 'r') as f:
        data = f.read().decode("utf-8-sig").encode("utf-8")
        return data.splitlines()


def load_labels(labels_file):
    label_strings = _load_file_as_array(labels_file)
    return np.array(map(int, label_strings))


def load_feature_matrix(matrix_file):
    with open(matrix_file) as f:
        data = [map(int, row) for row in csv.reader(f)]
    return np.array(data)


# Load features & labels.
train_features = load_feature_matrix(train_feature_matrix_file)
train_labels = load_labels(train_label_file)

test_features = load_feature_matrix(test_feature_matrix_file)
test_true_labels = load_labels(test_true_label_file)


def do_simple_mnb():
    mnb_clf = MultinomialNB()
    mnb_clf.fit(train_features, train_labels)

    train_predicted_labels = mnb_clf.predict(train_features)
    test_predicted_labels = mnb_clf.predict(test_features)

    y_score = mnb_clf.fit(train_features, train_labels).predict_proba(test_features)

    fpr, tpr, thresholds = roc_curve(test_true_labels, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Multinomial naive bayes AUC: {0}".format(roc_auc))

    print ("Train accuracy: {} -- Test accuracy: {}".format(
        mnb_clf.score(train_features, train_labels), mnb_clf.score(test_features, test_true_labels)
    ))

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def do_cv_mnb():
    print "CV"
    # Graph Stuff
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange']
    lw = 2
    i = 0

    # Perform cross validation
    cross_features = np.concatenate((train_features, test_features)) if concat_test_train else train_features
    cross_labels = np.concatenate((train_labels, test_true_labels)) if concat_test_train else train_labels
    row, col = np.shape(cross_features)
    skf = StratifiedKFold(n_splits=(row/block_size_k))
    for train_indices, test_indices in skf.split(cross_features, cross_labels):
        train_X, test_X = cross_features[train_indices], cross_features[test_indices]
        train_Y, test_Y = cross_labels[train_indices], cross_labels[test_indices]
        mnb_clf = MultinomialNB()
        #mnb_clf.fit(train_X, train_Y)

        y_score = mnb_clf.fit(train_X, train_Y).predict_proba(test_X)

        fpr, tpr, thresholds = roc_curve(test_Y, y_score[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        print ("Train accuracy: {} -- Test accuracy: {}".format(
            mnb_clf.score(train_X, train_Y), mnb_clf.score(test_X, test_Y)
        ))
        plt.plot(fpr, tpr, lw=lw, color=colors[i%6],
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')
    mean_tpr /= skf.get_n_splits(cross_features, cross_labels)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return


def svm_with_kernel():
    print "CV"
    # Graph Stuff
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange']
    lw = 2
    i = 0

    # Perform cross validation
    cross_features = np.concatenate((train_features, test_features)) if concat_test_train else train_features
    cross_labels = np.concatenate((train_labels, test_true_labels)) if concat_test_train else train_labels
    row, col = np.shape(cross_features)
    skf = StratifiedKFold(n_splits=(row / block_size_k))
    for train_indices, test_indices in skf.split(cross_features, cross_labels):
        train_X, test_X = cross_features[train_indices], cross_features[test_indices]
        train_Y, test_Y = cross_labels[train_indices], cross_labels[test_indices]

        clf = svm.SVC(kernel='linear', probability=True)
        # mnb_clf.fit(train_X, train_Y)

        y_score = clf.fit(train_X, train_Y).predict_proba(test_X)

        fpr, tpr, thresholds = roc_curve(test_Y, y_score[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        print ("Train accuracy: {} -- Test accuracy: {}".format(
            clf.score(train_X, train_Y), clf.score(test_X, test_Y)
        ))
        plt.plot(fpr, tpr, lw=lw, color=colors[i % 6],
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')
    mean_tpr /= skf.get_n_splits(cross_features, cross_labels)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return


def do_simple_svm(degree=1, do_plt=True):
    # svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    # rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    # poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    # lin_svc = svm.LinearSVC(C=C).fit(X, y)
    #
    clf = svm.SVC(kernel='poly', degree=degree, probability=True)
    clf.fit(train_features, train_labels)

    train_predicted_labels = clf.predict(train_features)
    test_predicted_labels = clf.predict(test_features)

    y_score = clf.fit(train_features, train_labels).predict_proba(test_features)

    fpr, tpr, thresholds = roc_curve(test_true_labels, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    print("AUC: {0}".format(roc_auc))

    print ("Train accuracy: {} -- Test accuracy: {}".format(
        clf.score(train_features, train_labels), clf.score(test_features, test_true_labels)
    ))

    if do_plt:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


def cross_validate_svm():
    i = 0.1
    while i < 1.0:
        do_simple_svm(i, do_plt=False)
        i += .1
    return


def simple_DT(do_plt=True):
    clf = DecisionTreeClassifier(criterion="gini")
    clf.fit(train_features, train_labels)

    train_predicted_labels = clf.predict(train_features)
    test_predicted_labels = clf.predict(test_features)

    y_score = clf.fit(train_features, train_labels).predict_proba(test_features)

    fpr, tpr, thresholds = roc_curve(test_true_labels, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    print("AUC: {0}".format(roc_auc))

    print ("Train accuracy: {} -- Test accuracy: {}".format(
        clf.score(train_features, train_labels), clf.score(test_features, test_true_labels)
    ))

    if do_plt:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

# simple_DT()


def simple_RF(do_plt=True):
    clf = RandomForestClassifier(n_estimators=20)
    clf.fit(train_features, train_labels)

    train_predicted_labels = clf.predict(train_features)
    test_predicted_labels = clf.predict(test_features)

    y_score = clf.fit(train_features, train_labels).predict_proba(test_features)

    fpr, tpr, thresholds = roc_curve(test_true_labels, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    print("AUC: {0}".format(roc_auc))

    print ("Train accuracy: {} -- Test accuracy: {}".format(
        clf.score(train_features, train_labels), clf.score(test_features, test_true_labels)
    ))

    if do_plt:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

#simple_RF()


def simple_KNN(do_plt=True):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(train_features, train_labels)

    train_predicted_labels = clf.predict(train_features)
    test_predicted_labels = clf.predict(test_features)

    y_score = clf.fit(train_features, train_labels).predict_proba(test_features)

    fpr, tpr, thresholds = roc_curve(test_true_labels, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    print("AUC: {0}".format(roc_auc))

    print ("Train accuracy: {} -- Test accuracy: {}".format(
        clf.score(train_features, train_labels), clf.score(test_features, test_true_labels)
    ))

    if do_plt:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


def simple_any(clf, do_plt=True):
    stats = dict()
    clf = clf.fit(train_features, train_labels)

    train_predicted_labels = clf.predict(train_features)
    test_predicted_labels = clf.predict(test_features)

    y_score = clf.predict_proba(test_features)

    stats['false_pv_arr'], stats['true_pv_arr'], thresholds = roc_curve(test_true_labels, y_score[:, 1])
    stats['precision_arr'], stats['recall_arr'], prthres = precision_recall_curve(test_true_labels, y_score[:, 1])
    stats['auc'] = auc(stats['false_pv_arr'], stats['true_pv_arr'])
    stats['aver_pr'] = average_precision_score(test_true_labels, y_score[:, 1])
    stats['precision_recall_f1'] = classification_report(test_true_labels, test_predicted_labels)
    stats['confusion_matrix'] = confusion_matrix(test_true_labels, test_predicted_labels)
    stats['test_accuracy'] = accuracy_score(test_true_labels, test_predicted_labels)
    stats['train_accuracy'] = accuracy_score(train_labels, train_predicted_labels)


    # print("AUC: {0}".format(roc_auc))

    # print ("Train accuracy: {} -- Test accuracy: {}".format(
    #     clf.score(train_features, train_labels), clf.score(test_features, test_true_labels)
    # ))


    # if do_plt:
    #     plt.figure()
    #     lw = 2
    #     plt.plot(fpr, tpr, color='darkorange',
    #              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    #     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver operating characteristic example')
    #     plt.legend(loc="lower right")
    #     plt.show()
    return stats


#simple_any(LogisticRegressionCV())
#simple_any(LogisticRegression())
#simple_any(AdaBoostClassifier())
#simple_any(GradientBoostingClassifier())

def run_classifier_with_stats():
    ada_stats = simple_any(AdaBoostClassifier())
    gbc_stats = simple_any(LogisticRegression())
    print ada_stats['train_accuracy']
    print ada_stats['test_accuracy']
    print ada_stats['confusion_matrix']
    print ada_stats['precision_recall_f1']
    plt.figure()
    lw = 2
    plt.plot(ada_stats['false_pv_arr'], ada_stats['true_pv_arr'],
             color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % ada_stats['auc'])
    plt.plot(gbc_stats['false_pv_arr'], gbc_stats['true_pv_arr'],
             color='blue', lw=lw, label='ROC curve (area = %0.2f)' % gbc_stats['auc'])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure()
    lw = 2
    plt.plot(ada_stats['recall_arr'], ada_stats['precision_arr'],
             color='darkorange', lw=lw, label='Average Precision (area = %0.2f)' % ada_stats['aver_pr'])
    plt.plot(gbc_stats['recall_arr'], gbc_stats['precision_arr'],
             color='blue', lw=lw, label='Average Precision (area = %0.2f)' % gbc_stats['aver_pr'])
    plt.axhline(y=0.5, color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall Rate')
    plt.ylabel('Precision Rate')
    plt.title('Precision Recall Curve')
    plt.legend(loc="lower right")
    plt.show()


run_classifier_with_stats()

def explore_knn(X_train, y_train):
    # here must be some code for your training set and test set
    #tuned_parameters = [{'n_neighbors': [1, 2, 3],
    #                     'weights': ['distance', 'uniform'],
    #                     'algorithm': ['ball_tree', 'kd_tree', 'brute']}]

    tuned_parameters = [{'n_neighbors': [1, 3, 5, 7, 9]}]

    scores = ['precision', 'recall']

    performances = []
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    # y_true, y_pred = y_test, clf.predict(X_test)
    # print(classification_report(y_true, y_pred))
    # print()


class DTSearch:
    def __init__(self):
        return

    @staticmethod
    def report(grid_scores, n_top=3):
        top_scores = sorted(grid_scores,
                            key=itemgetter(1),
                            reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print(("Mean validation score: "
                   "{0:.3f} (std: {1:.3f})").format(
                score.mean_validation_score,
                np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")

        return top_scores[0].parameters

    @staticmethod
    def run_gridsearch(X, y, clf, param_grid, cv=5):
        grid_search = GridSearchCV(clf,
                                   param_grid=param_grid,
                                   cv=cv)
        start = time()
        grid_search.fit(X, y)

        print(("\nGridSearchCV took {:.2f} "
               "seconds for {:d} candidate "
               "parameter settings.").format(time() - start,
                                             len(grid_search.grid_scores_)))

        top_params = DTSearch.report(grid_search.grid_scores_, 3)
        return top_params

    @staticmethod
    def do_it(X, y):
        features = ["SepalLength", "SepalWidth",
                    "PetalLength", "PetalWidth"]

        print("-- 10-fold cross-validation ")
        dt_old = DecisionTreeClassifier(min_samples_split=20,
                                        random_state=99)
        dt_old.fit(X, y)
        scores = cross_val_score(dt_old, X, y, cv=10)
        print("mean: {:.3f} (std: {:.3f})\n".format(scores.mean(),
                                                  scores.std()))

        print("-- Grid Parameter Search via 10-fold CV")

        # set of parameters to test
        param_grid = {"criterion": ["gini", "entropy"],
                      "min_samples_split": [2, 10, 20],
                      "max_depth": [None, 2, 5, 10],
                      "min_samples_leaf": [1, 5, 10],
                      "max_leaf_nodes": [None, 5, 10, 20],
                      }

        dt = DecisionTreeClassifier()
        ts_gs = DTSearch.run_gridsearch(X, y, dt, param_grid, cv=10)

        print("\n-- Best Parameters:")
        for k, v in ts_gs.items():
            print("parameter: {:<20s} setting: {}".format(k, v))

        # test the retuned best parameters
        print("\n\n-- Testing best parameters [Grid]...")
        dt_ts_gs = DecisionTreeClassifier(**ts_gs)
        scores = cross_val_score(dt_ts_gs, X, y, cv=10)
        print("mean: {:.3f} (std: {:.3f})\n\n".format(scores.mean(),
                                                  scores.std()))

        print("\n-- get_code for best parameters [Grid]:\n\n")
        dt_ts_gs.fit(X, y)

# DTSearch.do_it(train_features, train_labels)

