from pprint import pprint

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import functions
from functions import importOralToxicity

RANDOM_STATE = functions.RANDOM_STATE

data = functions.importHeartFailure()  # importOralToxicity
target = "DEATH_EVENT"  # 1024

estimators = {
    # 'KNN': KNeighborsClassifier(n_neighbors=9, metric='euclidean'),
    # 'DecisionTree': DecisionTreeClassifier(max_depth=5, criterion='gini', min_impurity_decrease=1.0e-02,
    #                                        random_state=RANDOM_STATE),
    # 'LinearRegression': LinearRegression(copy_X=True, n_jobs=-1),
    'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, solver='lbfgs', max_iter=500,
                                             multi_class='auto'),
    # 'RandomForest': RandomForestClassifier(n_estimators=75, max_depth=5, max_features=0.3,
    #                                        random_state=RANDOM_STATE),
    # 'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE)
}

estimatorParams = {
    # 'KNN': {
    #     'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    #     'metric': ['manhattan', 'euclidean', 'chebyshev']
    # },
    'DecisionTree': {
        'min_impurity_decrease': [0.025, 0.01, 0.005, 0.0025, 0.001],
        'max_depth': [2, 5, 10, 15, 20, 25],
        'criterion': ['entropy', 'gini']
    },
    'LinearRegression': {
        'copy_X': [True],
        'n_jobs': [-1],
        'fit_intercept': [False],
        'normalize': [True]
    },
    'LogisticRegression': {
        'C': [1e4, 6e3, 2e3,
              1000, 600, 200,
              100, 60, 20,
              10, 6, 5, 4, 3, 2,
              1, .9, .8, .7, .6, .5, .4, .3, .2,
              .1, .09, .08, .07, .06, .05, .04, .03, .02, .01],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }
    # 'RandomForest': {
    #     'n_estimators': [5, 10, 25, 50, 75, 100, 150, 200, 250, 300],
    #     'max_features': [.1, .3, .5, .7, .9, 1],
    #     'max_depth': [5, 10, 25]
    # },
    # 'GradientBoosting': {
    #     'n_estimators': [5, 10, 25, 50, 75, 100, 150, 200, 250, 300],
    #     'max_depth': [5, 10, 25],
    #     'learning_rate': [.1, .3, .5, .7, .9]
    # }
}

# Don't remove variables
correlationThresholds = [None]  # 0.85, 0.9, 0.95
varianceThresholds = [None]  # 0.85, 0.9, 0.95

# you should use feature extraction or feature selection, not both
featureExtractionVariances = [None]  # 0.7, 0.75, 0.85, 0.9, 0.95

featureSelectionMethods = [None]  # 'chi2', 'anova', None
numFeatures = [50, 100, 300, 500]  # 50, 100, 300, 500

scalingMethods = [None]  # 'minmax', 'z-score', None
# 'undersampling', 'oversampling', 'smote', None
balancingMethods = ['undersampling', 'oversampling', 'smote', None]

best, allResults = functions.findBest(data, target, correlationThresholds, varianceThresholds, featureExtractionVariances,
                                      featureSelectionMethods, numFeatures, scalingMethods, balancingMethods,
                                      estimators, estimatorParams, omitErrors=False, n_jobs=4, verbose=1, graph_confusion_matrix=False)

#pprint(allResults)
functions.graphResults(allResults, compare="scalingMethod")
acc = best.pop('accuracy')
print("Best results combination {} -> accuracy: {} ".format(best, acc))
