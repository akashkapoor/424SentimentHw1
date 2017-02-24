from itertools import cycle

from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

import csv
import matplotlib.pyplot as plt
import numpy as np

# K-Fold Constants
concat_test_train = True
block_size_k = 10

# File names
vocab_file = 'out_vocab_3.txt'
train_label_file = 'out_classes_3.txt'
train_feature_matrix_file = 'out_bag_of_words_3.csv'

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


def do_single():
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


def do_cv():
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

do_cv()