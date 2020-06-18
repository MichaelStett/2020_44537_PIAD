from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc, make_scorer

from matplotlib.colors import ListedColormap

from timeit import default_timer as timer

import matplotlib.pyplot as plt
import pandas as pd
import random as rd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

### ZADANIE 1
## 1
#X, Y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_classes=2, n_clusters_per_class=2, random_state=2137)

## 2
#plt.scatter(X[:, 0], X[:, 1], 35, Y, 'o');
#plt.show();

def classify(clf, X, Y, seed = 2137, iter = 100):

    acs = []; rcs = []; pcs = []; fcs = []; racs = []; ltimes = []; ttimes = []

    for i in range(iter):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed)

        learnStart = timer()
        clf = clf.fit(X_train, y_train)
        learnStop = timer()

        predStart = timer()
        y_pred = clf.predict(X_test)
        predStop = timer()

        y_true = y_test

        acs.append(accuracy_score(y_true, y_pred))     # dokładność
        rcs.append(recall_score(y_true, y_pred))       # czułość
        pcs.append(precision_score(y_true, y_pred))    # precyzja
        fcs.append(f1_score(y_true, y_pred))           # F1?
        racs.append(roc_auc_score(y_true, y_pred))     # pole pod krzywą roc
        ltimes.append((learnStop - learnStart) * 400)
        ttimes.append((predStop - predStart) * 400)

        #if i == iter - 1: # last
        #    #plotDiff(X_train, X_test, y_train, y_test, y_pred)
        #    #plotROC(y_test, y_pred)
        #    plotDiscrimination(X, Y, clf)

    ac_mean = np.mean(acs)
    rc_mean = np.mean(rcs)
    pc_mean = np.mean(pcs)
    fc_mean = np.mean(fcs)
    rac_mean = np.mean(racs)
    ltime_mean = np.mean(ltimes)
    ttime_mean = np.mean(ttimes)

    return [ac_mean, rc_mean, pc_mean, fc_mean, rac_mean, ltime_mean, ttime_mean];


def plotDiff(X_train, X_test, y_train, y_test, y_pred):

    X = np.concatenate((X_train, X_test))
    Y_TEST = np.concatenate((y_train, y_test))
    Y_PRED = np.concatenate((y_train, y_pred))

    Y_DIFF = Y_TEST == Y_PRED

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    ax1.set_title("oczekiwane")
    ax1.scatter(X[:, 0], X[:, 1], 25, Y_PRED, 'o')

    ax2.set_title("obliczone")
    ax2.scatter(X[:, 0], X[:, 1], 25, Y_TEST, 'o')

    ax3.set_title("różnice")
    ax3.scatter(X[:, 0], X[:, 1], 25, Y_DIFF, 'o')

    plt.show();


def plotROC(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr, label=f'AUC: {round(roc_auc, 2)}');
    plt.legend(loc="lower right");
    plt.show();


def plotDiscrimination(X, Y, clf):
    size = 100

    xMin, xMax = X[:, 0].min(), X[:, 0].max()
    yMin, yMax = X[:, 1].min(), X[:, 1].max()

    xx = np.linspace(xMin, xMax, size)
    yy = np.linspace(yMin, yMax, size)
    
    XX, YY = np.meshgrid(xx, yy)
    points = np.vstack((XX.flatten(), YY.flatten())).transpose()

    x_labels = clf.predict(points).reshape(size, size)

    plt.xlim(xMin, xMax)
    plt.ylim(yMin, yMax)
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.contour(
        points[:,0].reshape(size, size), 
        points[:,1].reshape(size, size), 
        x_labels)
    plt.show();

# 3
clfs = [
    KNeighborsClassifier(),
    QuadraticDiscriminantAnalysis(),
    SVC(probability=True),
    DecisionTreeClassifier(),
    GaussianNB()
]

index = [
    'accuracy_score', 
    'recall_score', 
    'precision_score', 
    'f1_score', 
    'roc_auc_score', 
    'train_time', 
    'test_time'
]

# data = [classify(clf, X, Y) for clf in clfs]

#data_clfs = {
#    'KNeighbors': data[0],
#    'Quadratic': data[1],
#    'SVC': data[2],
#    'DecisionTree': data[3],
#    'GaussianNB': data[4]
#}

# pd.DataFrame(data_clfs, index).plot(kind='bar', grid=False); plt.show();

#### ZADANIE 2

## 1
data = make_classification(n_samples=200)
XY = data[0]
X = data[0][:, 0]
Y = data[0][:, 1]
Z = data[1]

# 2 wybrano SVC
print(X.shape)
print(Y.shape)

# 4
gr = GridSearchCV(
    estimator = SVC(probability=True),
    scoring='accuracy',
    cv=5,
    param_grid = {
        'C': np.arange(0, 20, 5),
        'kernel': ('linear', 'rbf'),
        #'gamma': ('scale', 'auto'),
})

gr.fit(XY, Z)

res = pd.DataFrame(gr.cv_results_)

plt.plot(res['mean_test_score'].values); plt.show()

