import itertools
import sys
from pprint import pprint

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy.spatial.distance import squareform, pdist
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.cluster import adjusted_rand_score, fowlkes_mallows_score
from utils import ds_functions as ds
import matplotlib.pyplot as plt


RANDOM_STATE = 42


def importHeartFailure():
    data = pd.read_csv(
        'data/heart_failure_clinical_records_dataset.csv', sep=",", decimal=".")

    for col in ['anaemia', 'diabetes', 'high_blood_pressure', 'smoking', 'DEATH_EVENT', 'sex']:
        data[col] = data[col].apply(lambda x: bool(x))

    return data


def importOralToxicity():
    data = pd.read_csv('data/qsar_oral_toxicity.csv', sep=";", header=None)

    data[1024] = data[1024].replace(['positive', 'negative'], [True, False])

    return data


def imputeOutliers(data: pd.DataFrame, threshold: int = 3, printNum=False) -> pd.DataFrame:
    if not threshold:
        return
    to_delete = {}
    for col in data.select_dtypes(include='number').columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        for index, value in data[col].iteritems():
            if value > high or value < low:
                if index in to_delete:
                    to_delete[index] += 1
                else:
                    to_delete[index] = 1
    deleted = 0
    for index, count in to_delete.items():
        if count >= threshold:
            data.drop(index=index)
            deleted += 1
    if printNum:
        print(deleted)
    return data


def missingValueImputation(df):
    cols_nr = df.select_dtypes(include='number')
    df_sb = df[df.select_dtypes(include=['category', 'bool']).columns]

    imp_nr = SimpleImputer(strategy='mean', missing_values=np.nan, copy=True)
    df_nr = pd.DataFrame(imp_nr.fit_transform(
        cols_nr), columns=cols_nr.columns)

    return df_nr.join(df_sb, how='right')


def scaling(df, method=None):
    """Scale dataset

    method (string): can be minmax, z-score or None
    """
    df_nr = df[df.select_dtypes(include='number').columns]
    df_sb = df[df.select_dtypes(include=['category', 'bool']).columns]

    if method == 'minmax':
        transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
        mm_df_nr = pd.DataFrame(transf.transform(
            df_nr), columns=df_nr.columns, index=df_nr.index)
        df = mm_df_nr.join(df_sb, how='right')
    elif method == 'z-score':
        transf = StandardScaler(
            with_mean=True, with_std=True, copy=True).fit(df_nr)
        z_df_nr = pd.DataFrame(transf.transform(
            df_nr), columns=df_nr.columns, index=df_nr.index)
        df = z_df_nr.join(df_sb, how='right')

    elif method is not None:
        print("Method \'{}\' not supported for scaling!".format(method))
        sys.exit(1)

    return df


def balancing(data, target, method=None, train_size=0.7):
    """Balance dataset and split into training and testing sets

    method (string): can be smote, undersampling, oversampling or None
    """
    df = data.copy()
    y: np.ndarray = df.pop(target).values
    X: np.ndarray = df.values

    _, tstX, _, tstY = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=RANDOM_STATE)

    if method == "SMOTE" or method == "smote":
        smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
        X, y = smote.fit_sample(X, y)

    elif method == "undersampling":
        undersampler = RandomUnderSampler(random_state=RANDOM_STATE)
        X, y = undersampler.fit_sample(X, y)

    elif method == "oversampling":
        oversampler = RandomOverSampler(random_state=RANDOM_STATE)
        X, y = oversampler.fit_sample(X, y)

    elif method is not None:
        print("Method \'{}\' not supported for balancing!".format(method))
        sys.exit(1)

    trnX, _, trnY, _ = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=RANDOM_STATE)

    values = {'Train': [trnX, trnY], 'Test': [tstX, tstY], "y": y}

    return values


def applyEstimator(estimator, values, omitErrors=False):
    """Fit estimator with training set and return accuracy predicted in test set
    """
    try:
        estimator.fit(values['Train'][0], values['Train'][1])
        prdY = estimator.predict(values['Test'][0])
        #prdY = [False if p < 0 else True for p in prdY]
        return metrics.accuracy_score(prdY, values['Test'][1]), metrics.f1_score(values['Test'][1], prdY)
    except Exception as e:
        if not omitErrors:
            print("Oops!", e.__class__, "occurred while applying estimator.")
            print("Skipping estimator")
        return -1, -1


def findBestParams(estimator, params, values, n_jobs=-1, verbose=1):
    """Applies grid search to determine best params for estimator
    Return best estimator and corresponding parameters
    """
    search = GridSearchCV(estimator, params, cv=3,
                          n_jobs=n_jobs, verbose=verbose, scoring=metrics.make_scorer(metrics.f1_score))
    search.fit(values['Train'][0], values['Train'][1])
    return search.best_estimator_, search.best_params_


def featureSelection(values, method, k, omitErrors=False):
    """Selects k best features

    method (string): can be chi2, anova or None
    k (int): number of features to select
    """
    try:
        if method == "chi2":
            score_func = chi2
        elif method == "anova" or method == "ANOVA":
            score_func = f_classif
        elif method is None:
            return values
        else:
            print("Method \'{}\' not supported for feature selection!".format(method))
            sys.exit(1)

        selector = SelectKBest(score_func, k=k)
        values['Train'][0] = selector.fit_transform(
            values['Train'][0], values['Train'][1])
        values['Test'][0] = selector.fit(values['Test'][0], values['Test'][1])

        return values

    except Exception as e:
        print(e)
        if not omitErrors:
            print("Oops!", e.__class__,
                  "occurred while selecting features using {}.".format(method))
            print("Skipping feature selection")
        return values


def featureExtraction(data, target, variance, minimum2=False):
    targetData = data.pop(target)
    mean = (data.mean(axis=0)).tolist()
    centered_data = data - mean

    pca = PCA(n_components=variance, svd_solver='full')
    pca.fit(centered_data)

    transf = pca.transform(data)

    if minimum2 and len(transf[0]) < 2:
        pca = PCA(n_components=2)
        pca.fit(centered_data)

        transf = pca.transform(data)

    data = pd.DataFrame(transf)
    return data.join(targetData)


def correlation_removal(data: pd.DataFrame, target, threshold: float) -> pd.DataFrame:
    data = data.copy(deep=True)
    y = data.pop(target)
    col_corr = set()
    corr_matrix = data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                if colname in data.columns:
                    del data[colname]  # deleting the column from the dataset
    data[target] = y
    return data


def remove_low_variance(data: pd.DataFrame, target, threshold: float) -> pd.DataFrame:
    data = data.copy(deep=True)
    y = data.pop(target)
    x = data.values
    sel = VarianceThreshold(threshold=threshold * (1 - threshold))
    x = sel.fit_transform(x)
    data = pd.DataFrame(x)
    data[target] = y
    return data


def balance_and_split(data: pd.DataFrame, target, splitting_strategy: str, balancing_strategy: str, *,
                      random_state: int = 42):
    # data = data.copy(deep=True) Not needed when threading
    y = data.pop(target).values
    x = data.values
    data[target] = y
    if splitting_strategy == "holdout":
        trn_x, tst_x, trn_y, tst_y = train_test_split(
            x, y, train_size=0.7, stratify=y, random_state=random_state)
        if balancing_strategy == "undersampling":
            sampler = RandomUnderSampler(random_state=random_state)
            trn_x_balanced, trn_y_balanced = sampler.fit_sample(trn_x, trn_y)
        elif balancing_strategy == "oversampling":
            sampler = RandomOverSampler(random_state=random_state)
            trn_x_balanced, trn_y_balanced = sampler.fit_sample(trn_x, trn_y)
        elif balancing_strategy == "smote":
            sampler = SMOTE(sampling_strategy='minority',
                            random_state=random_state)
            trn_x_balanced, trn_y_balanced = sampler.fit_sample(trn_x, trn_y)
        else:
            raise Exception(
                f"Unsupported balancing strategy {balancing_strategy}")
        yield trn_x_balanced, tst_x, trn_y_balanced, tst_y
    elif splitting_strategy == "cross-validation":
        splits = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=random_state).split(x, y)
        for train_index, test_index in splits:
            trn_x, trn_y = x[train_index], y[train_index]
            tst_x, tst_y = x[test_index], y[test_index]
            if balancing_strategy == "undersampling":
                sampler = RandomUnderSampler(random_state=random_state)
                trn_x_balanced, trn_y_balanced = sampler.fit_sample(
                    trn_x, trn_y)
            elif balancing_strategy == "oversampling":
                sampler = RandomOverSampler(random_state=random_state)
                trn_x_balanced, trn_y_balanced = sampler.fit_sample(
                    trn_x, trn_y)
            elif balancing_strategy == "smote":
                sampler = SMOTE(sampling_strategy='minority',
                                random_state=random_state)
                trn_x_balanced, trn_y_balanced = sampler.fit_sample(
                    trn_x, trn_y)
            else:
                raise Exception(
                    f"Unsupported balancing strategy {balancing_strategy}")
            yield trn_x_balanced, tst_x, trn_y_balanced, tst_y
    else:
        raise Exception(f"Invalid splitting strategy {splitting_strategy}")


# def train_kfold():
#     for name, classifier in classifiers.items():
#         xvalues += [name]
#         sets = balance_and_split(data, target, splitting_strategy, balancing_strategy,
#                                  random_state=random_state)

#         kfold_results = [[], []]
#         for trn_x, tst_x, trn_y, tst_y in sets:
#             nvb = classifier(random_state=random_state)
#             try:
#                 nvb.fit(trn_x, trn_y)
#             except ValueError as e:
#                 print(repr(e))
#                 xvalues.pop()
#                 break
#             prd_y = nvb.predict(tst_x)
#             score = metrics.accuracy_score(tst_y, prd_y)
#             kfold_results[0] += [(nvb, score, trn_x, tst_x, trn_y, tst_y)]
#             kfold_results[1] += [score]
#         else:
#             average_score = np.average(kfold_results[1])
#             if average_score > best[0]:
#                 best = (average_score, name, None)
#                 diff = 1
#                 for (clf, score, trn_x, tst_x, trn_y, tst_y) in kfold_results[0]:
#                     if abs(average_score - score) < diff:
#                         diff = average_score - score
#                         best = (average_score, name, clf)
#                         best_set = [trn_x, tst_x, trn_y, tst_y]
#             yvalues += [average_score]

        # plt.figure()
        # ds.bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
        # plt.show()

def graphConfusionMatrix(estimator, values, omitErrors=False):
    """Fit estimator with training set and return accuracy predicted in test set
    """

    try:
        estimator.fit(values['Train'][0], values['Train'][1])
        prdY = estimator.predict(values['Test'][0])
        #prdY = [False if p < 0 else True for p in prdY]

    except Exception as e:
        if not omitErrors:
            print("Oops!", e.__class__, "occurred while graphing estimator.")
            print("Skipping confusion matrix")
        return -1

    labels: np.ndarray = pd.unique(values['y'])
    cnf_mtx: np.ndarray = metrics.confusion_matrix(
        values['Test'][1], prdY, labels)

    def plot_confusion_matrix(cnf_matrix: np.ndarray, classes_names: np.ndarray, ax: plt.Axes = None,
                              normalize: bool = False):
        CMAP = plt.cm.Blues

        if ax is None:
            ax = plt.gca()
        if normalize:
            total = cnf_matrix.sum(axis=1)[:, np.newaxis]
            cm = cnf_matrix.astype('float') / total
            title = "Normalized confusion matrix"
        else:
            cm = cnf_matrix
            title = 'Confusion matrix'
        np.set_printoptions(precision=2)
        tick_marks = np.arange(0, len(classes_names), 1)
        ax.set_title(title)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes_names)
        ax.set_yticklabels(classes_names)
        ax.imshow(cm, interpolation='nearest', cmap=CMAP)

        fmt = '.2f' if normalize else 'd'
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                    color='w', horizontalalignment="center")

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
    plot_confusion_matrix(cnf_mtx, labels, ax=axs[0, 0])
    plot_confusion_matrix(metrics.confusion_matrix(
        values['Test'][1], prdY, labels), labels, axs[0, 1], normalize=True)
    plt.tight_layout()
    plt.show()


def graphResults(results, compare="estimatorName"):
    ''' 
    'compare' value will divide bars in different charts. 
    Example:
    If there are 6 estimators, a combination of 3 scaling methods and 2 balancing methods
    Then compare='balancingMethod' (singular, as appears in the dictionary)
    will cause the function to plot 2 figures, one for each balancingMethod.
    '''

    sets = []
    divisors = set([r[compare] for r in results])
    print(compare)
    print(divisors)
    for d in divisors:
        sets.append([r for r in results if r[compare] == d])
    results = sets

    _, axs = plt.subplots(len(results), sharey=True)

    if len(results) == 1:
        axs = [axs]

    for ax_i, subset in enumerate(results):
        yvalues = {'accuracy': [],
                   'f1': []
                   }
        xvalues = []
        for i, r in enumerate(subset):
            yvalues['accuracy'].append(r['accuracy'])
            yvalues['f1'].append(r['f1'])
            xvalues.append(f"{i}")
            print(f"{i}: {r}\n")

        # ds.bar_chart(xvalues, yvalues, title=str(r[compare]),
        #              ylabel='accuracy', ax=axs[ax_i], percentage=True)
        ds.multiple_bar_chart(xvalues, yvalues, title=str(r[compare]),
                              xlabel='estimators', ylabel='metrics', ax=axs[ax_i], percentage=True)

    plt.show()


def getCombs(a, b):
    combs = itertools.product(a, b)
    combs = [comb for comb in combs]
    if None in [comb[0] for comb in combs]:
        combs = [comb for comb in combs if comb[0] is not None]
        combs.append((None, 0))
    return combs


def findBest(data, target, correlationThresholds, varianceThresholds, featureExtractionVariances, featureSelectionMethods, numFeatures, scalingMethods,
             balancingMethods, estimators, estimatorParams, omitErrors=False, n_jobs=-1, verbose=1, graph_confusion_matrix=False):
    """Finds best combination of methods, estimators and parameteres for dataset

    data: dataset
    target: target variable
    featureSelectionMethods: list of feature selection methods to try
    numFeatures: list of number of features to select using feature selection
    scalingMethods: list of scaling methods to try
    balancingMethods: list of balancing methods to try
    estimators: dictionary with estimator name as key and estimator as value
    estimatorParams: dictionary with estimator name as key and dictionary of parameters as value
    omitErrors (bool): ignore errors (for example negative values using chi2 for feature selection)
    n_jobs (int): number of cores to use in grid search (-1 = all cores)
    verbose (int): amount of messages to send during grid search (0 to 10)
    """
    bestAcc = 0
    best = []
    allResults = []
    print("Trying", list(getCombs(featureSelectionMethods, numFeatures)),
          "combinations from", featureSelectionMethods, "and", numFeatures)
    for correlationThreshold in correlationThresholds:
        if correlationThreshold is not None:
            correlation_removal(data, target, correlationThreshold)
        for varianceThreshold in varianceThresholds:
            if varianceThreshold is not None:
                remove_low_variance(data, target, varianceThreshold)
            for featureExtractionVariance in featureExtractionVariances:
                if featureExtractionVariance is not None:
                    data = featureExtraction(
                        data, target, featureExtractionVariance)
                for scalingMethod in scalingMethods:
                    data = scaling(data, scalingMethod)
                    for balancingMethod in balancingMethods:
                        values = balancing(data, target, balancingMethod)
                        for featureSelectionMethod, k in getCombs(featureSelectionMethods, numFeatures):
                            values = featureSelection(
                                values, featureSelectionMethod, k, omitErrors)
                            for estimatorName, estimator in estimators.items():
                                bestParams = 'default'
                                bestEstimator = estimator

                                if bool(estimatorParams[estimatorName]):
                                    bestEstimator, bestParams = findBestParams(estimator, estimatorParams[estimatorName], values,
                                                                               n_jobs=n_jobs, verbose=verbose)

                                acc, f1 = applyEstimator(
                                    bestEstimator, values, omitErrors)

                                combination = {
                                    'estimatorName': estimatorName,
                                    'bestParams': bestParams,
                                    'correlationThreshold': correlationThreshold,
                                    'varianceThreshold': varianceThreshold,
                                    'featureExtractionVariance': featureExtractionVariance,
                                    'featureSelectionMethod': featureSelectionMethod,
                                    'numFeatures': k,
                                    'scalingMethod': scalingMethod,
                                    'balancingMethod': balancingMethod,
                                    'accuracy': acc,
                                    'f1': f1
                                }

                                if acc > bestAcc:
                                    bestAcc = acc
                                    best = combination

                                allResults.append(combination)

                                if graph_confusion_matrix:
                                    graphConfusionMatrix(estimator, values)

    return best, allResults

# Unsupervised Learning

def kmeans(data, v1, v2, N_CLUSTERS, rows, cols, compareToTargetVerbose=0, targetData=None, similarityThreshold=0):
    mse: list = []
    sc: list = []
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
    i, j = 0, 0
    maxSim = {"max": -np.inf, "cluster": -1, "k": -1}
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = KMeans(n_clusters=k, random_state=RANDOM_STATE)
        estimator.fit(data)
        mse.append(estimator.inertia_)
        sc.append(silhouette_score(data, estimator.labels_))
        ds.plot_clusters(data, v2, v1, estimator.labels_.astype(float), estimator.cluster_centers_, k,
                         f'KMeans k={k}', ax=axs[i, j])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

        if compareToTargetVerbose > 0:
            if compareToTargetVerbose > 1:
                print(k, "clusters:")
            for cluster in range(k):
                score = adjusted_rand_score(targetData, np.array(estimator.labels_) == cluster)
                if compareToTargetVerbose > 1 and score > similarityThreshold:
                    print("Cluster", cluster, "->", score)
                if maxSim["max"] < score:
                    maxSim["max"] = score
                    maxSim["cluster"] = i
                    maxSim["k"] = k

    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
    ds.plot_line(N_CLUSTERS, mse, title='KMeans MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
    ds.plot_line(N_CLUSTERS, sc, title='KMeans SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
    plt.ylim(-1, 1)
    plt.show()

    if compareToTargetVerbose > 0:
        print(maxSim)


def em(data, v1, v2, N_CLUSTERS, rows, cols, compareToTargetVerbose=0, targetData=None, similarityThreshold=0):
    mse: list = []
    sc: list = []
    _, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
    i, j = 0, 0
    maxSim = {"max": -np.inf, "cluster": -1, "k": -1}
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
        estimator.fit(data)
        labels = estimator.predict(data)
        mse.append(ds.compute_mse(data.values, labels, estimator.means_))
        sc.append(silhouette_score(data, labels))
        ds.plot_clusters(data, v2, v1, labels.astype(float), estimator.means_, k,
                         f'EM k={k}', ax=axs[i, j])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

        if compareToTargetVerbose > 0:
            if compareToTargetVerbose > 1:
                print(k, "clusters:")
            for cluster in range(k):
                score = adjusted_rand_score(targetData, np.array(labels) == cluster)
                if compareToTargetVerbose > 1 and score > similarityThreshold:
                    print("Cluster", cluster, "->", score)
                if maxSim["max"] < score:
                    maxSim["max"] = score
                    maxSim["cluster"] = i
                    maxSim["k"] = k

    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
    ds.plot_line(N_CLUSTERS, mse, title='EM MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
    ds.plot_line(N_CLUSTERS, sc, title='EM SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
    plt.ylim(-1, 1)
    plt.show()

    if compareToTargetVerbose > 0:
        print(maxSim)


def eps(data, v1, v2, compareToTargetVerbose=0, targetData=None, similarityThreshold=0):
    EPS = [2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    mse: list = []
    sc: list = []
    rows, cols = ds.choose_grid(len(EPS))
    _, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
    i, j = 0, 0
    maxSim = {"max": -np.inf, "cluster": -1, "k": -1, "eps": -1}
    for n in range(len(EPS)):
        estimator = DBSCAN(eps=EPS[n], min_samples=2)
        estimator.fit(data)
        labels = estimator.labels_
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k > 1:
            centers = ds.compute_centroids(data, labels)
            mse.append(ds.compute_mse(data.values, labels, centers))
            sc.append(silhouette_score(data, labels))
            ds.plot_clusters(data, v2, v1, labels.astype(float), estimator.components_, k,
                             f'DBSCAN eps={EPS[n]} k={k}', ax=axs[i, j])
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

            if compareToTargetVerbose > 0:
                if compareToTargetVerbose > 1:
                    print(k, "clusters:")
                for cluster in range(k):
                    score = adjusted_rand_score(targetData, np.array(estimator.labels_) == cluster)
                    if compareToTargetVerbose > 1 and score > similarityThreshold:
                        print("Cluster", cluster, "->", score)
                    if maxSim["max"] < score:
                        maxSim["max"] = score
                        maxSim["cluster"] = i
                        maxSim["k"] = k
                        maxSim["eps"] = EPS[n]

        else:
            mse.append(0)
            sc.append(0)
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
    ds.plot_line(EPS, mse, title='DBSCAN MSE', xlabel='eps', ylabel='MSE', ax=ax[0, 0])
    ds.plot_line(EPS, sc, title='DBSCAN SC', xlabel='eps', ylabel='SC', ax=ax[0, 1], percentage=True)
    plt.ylim(-1, 1)
    plt.show()

    if compareToTargetVerbose > 0:
        print(maxSim)


def metric(data, v1, v2, compareToTargetVerbose=0, targetData=None, similarityThreshold=0):
    METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']
    distances = []
    for m in METRICS:
        dist = np.mean(np.mean(squareform(pdist(data.values, metric=m))))
        distances.append(dist)

    print('AVG distances among records', distances)
    distances[0] *= 0.6
    distances[1] = 80
    distances[2] *= 0.6
    distances[3] *= 0.1
    distances[4] *= 0.15
    print('CHOSEN EPS', distances)

    mse: list = []
    sc: list = []
    rows, cols = ds.choose_grid(len(METRICS))
    _, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
    i, j = 0, 0
    maxSim = {"max": -np.inf, "cluster": -1, "k": -1, "eps": -1, "metric": None}
    for n in range(len(METRICS)):
        estimator = DBSCAN(eps=distances[n], min_samples=2, metric=METRICS[n])
        estimator.fit(data)
        labels = estimator.labels_
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k > 1:
            centers = ds.compute_centroids(data, labels)
            mse.append(ds.compute_mse(data.values, labels, centers))
            sc.append(silhouette_score(data, labels))
            ds.plot_clusters(data, v2, v1, labels.astype(float), estimator.components_, k,
                             f'DBSCAN metric={METRICS[n]} eps={distances[n]:.2f} k={k}', ax=axs[i, j])

            if compareToTargetVerbose > 0:
                if compareToTargetVerbose > 1:
                    print(k, "clusters:")
                for cluster in range(k):
                    score = adjusted_rand_score(targetData, np.array(estimator.labels_) == cluster)
                    if compareToTargetVerbose > 1 and score > similarityThreshold:
                        print("Cluster", cluster, "->", score)
                    if maxSim["max"] < score:
                        maxSim["max"] = score
                        maxSim["cluster"] = i
                        maxSim["k"] = k
                        maxSim["eps"] = distances[n]
                        maxSim["metric"] = METRICS[n]

        else:
            mse.append(0)
            sc.append(0)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
    ds.bar_chart(METRICS, mse, title='DBSCAN MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
    ds.bar_chart(METRICS, sc, title='DBSCAN SC', xlabel='metric', ylabel='SC', ax=ax[0, 1], percentage=True)
    plt.ylim(-1, 1)
    plt.show()

    if compareToTargetVerbose > 0:
        print(maxSim)


def hierarchical(data, v1, v2, N_CLUSTERS, compareToTargetVerbose=0, targetData=None, similarityThreshold=0):
    mse: list = []
    sc: list = []
    rows, cols = ds.choose_grid(len(N_CLUSTERS))
    _, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
    i, j = 0, 0
    maxSim = {"max": -np.inf, "cluster": -1, "k": -1}
    for n in range(len(N_CLUSTERS)):
        k = N_CLUSTERS[n]
        estimator = AgglomerativeClustering(n_clusters=k)
        estimator.fit(data)
        labels = estimator.labels_
        centers = ds.compute_centroids(data, labels)
        mse.append(ds.compute_mse(data.values, labels, centers))
        sc.append(silhouette_score(data, labels))
        ds.plot_clusters(data, v2, v1, labels, centers, k,
                         f'Hierarchical k={k}', ax=axs[i, j])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

        if compareToTargetVerbose > 0:
            if compareToTargetVerbose > 1:
                print(k, "clusters:")
            for cluster in range(k):
                score = adjusted_rand_score(targetData, np.array(estimator.labels_) == cluster)
                if compareToTargetVerbose > 1 and score > similarityThreshold:
                    print("Cluster", cluster, "->", score)
                if maxSim["max"] < score:
                    maxSim["max"] = score
                    maxSim["cluster"] = i
                    maxSim["k"] = k

    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
    ds.plot_line(N_CLUSTERS, mse, title='Hierarchical MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
    ds.plot_line(N_CLUSTERS, sc, title='Hierarchical SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
    plt.ylim(-1, 1)
    plt.show()

    if compareToTargetVerbose > 0:
        print(maxSim)


def hierarchicalMetrics(data, v1, v2, compareToTargetVerbose=0, targetData=None, similarityThreshold=0):
    METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']
    LINKS = ['complete', 'average']
    k = 13
    values_mse = {}
    values_sc = {}
    rows = len(METRICS)
    cols = len(LINKS)
    _, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
    maxSim = {"max": -np.inf, "cluster": -1, "k": -1, "linkage": None, "metric": None}
    for i in range(len(METRICS)):
        mse: list = []
        sc: list = []
        m = METRICS[i]
        for j in range(len(LINKS)):
            link = LINKS[j]
            estimator = AgglomerativeClustering(n_clusters=k, linkage=link, affinity=m)
            estimator.fit(data)
            labels = estimator.labels_
            centers = ds.compute_centroids(data, labels)
            mse.append(ds.compute_mse(data.values, labels, centers))
            sc.append(silhouette_score(data, labels))
            ds.plot_clusters(data, v2, v1, labels, centers, k,
                             f'Hierarchical k={k} metric={m} link={link}', ax=axs[i, j])

            if compareToTargetVerbose > 0:
                if compareToTargetVerbose > 1:
                    print(k, "clusters:")
                for cluster in range(k):
                    score = adjusted_rand_score(targetData, np.array(estimator.labels_) == cluster)
                    if compareToTargetVerbose > 1 and score > similarityThreshold:
                        print("Cluster", cluster, "->", score)
                    if maxSim["max"] < score:
                        maxSim["max"] = score
                        maxSim["cluster"] = i
                        maxSim["k"] = k
                        maxSim["linkage"] = link
                        maxSim["metric"] = m

        values_mse[m] = mse
        values_sc[m] = sc
    plt.show()

    _, ax = plt.subplots(1, 2, figsize=(6, 3), squeeze=False)
    ds.multiple_bar_chart(LINKS, values_mse, title=f'Hierarchical MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
    ds.multiple_bar_chart(LINKS, values_sc, title=f'Hierarchical SC', xlabel='metric', ylabel='SC', ax=ax[0, 1],
                          percentage=True)
    plt.ylim(-1, 1)
    plt.show()

    if compareToTargetVerbose > 0:
        print(maxSim)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def fullClustering(data, target, v1=0, v2=1, N_CLUSTERS=(2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29),
                   withFeatureExtraction=False, variance=0.85, plotTarget=True, compareToTargetVerbose=0,
                   similarityThreshold=0):

    if withFeatureExtraction:
        transf = featureExtraction(data, target, variance, minimum2=True)
        # data = transf[[transf.columns[v1], transf.columns[v2]]]
        data = transf
        v1 = 1
        v2 = 0
    #     targetData = data.pop(target)
    # else:
    targetData = data.pop(target)

    if plotTarget:
        plt.title("var%s x var%s" % (v1, v2))
        plt.xlabel("var%s" % v1)
        plt.ylabel("var%s" % v2)
        points = plt.scatter(data.iloc[:, v2], data.iloc[:, v1], c=targetData, cmap=plt.cm.RdYlGn)
        plt.legend(*points.legend_elements(), title=target)
        plt.show()

    rows, cols = ds.choose_grid(len(N_CLUSTERS))
    print("knn")
    kmeans(data, v1, v2, N_CLUSTERS, rows, cols, compareToTargetVerbose, targetData, similarityThreshold)

    print("em")
    em(data, v1, v2, N_CLUSTERS, rows, cols, compareToTargetVerbose, targetData, similarityThreshold)

    print("eps")
    eps(data, v1, v2, compareToTargetVerbose, targetData, similarityThreshold)

    print("metric")
    metric(data, v1, v2, compareToTargetVerbose, targetData, similarityThreshold)

    print("hierarchical")
    hierarchical(data, v1, v2, N_CLUSTERS, compareToTargetVerbose, targetData, similarityThreshold)

    print("hierarchicalMetrics")
    hierarchicalMetrics(data, v1, v2, compareToTargetVerbose, targetData, similarityThreshold)