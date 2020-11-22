import itertools
import sys

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

RANDOM_STATE = 42


def importHeartFailure():
    data = pd.read_csv('data/heart_failure_clinical_records_dataset.csv', sep=",", decimal=".")

    for col in ['anaemia', 'diabetes', 'high_blood_pressure', 'smoking', 'DEATH_EVENT', 'sex']:
        data[col] = data[col].apply(lambda x: bool(x))

    return data


def importOralToxicity():
    data = pd.read_csv('data/qsar_oral_toxicity.csv', sep=";", header=None)

    data[1024] = data[1024].replace(['positive', 'negative'], [True, False])

    return data


# def imputeOutliers(df):
#     for col in df.select_dtypes(include='number').columns:
#         q1 = df[col].quantile(0.25)
#         q3 = df[col].quantile(0.75)
#         iqr = q3 - q1
#         low = q1 - 1.5 * iqr
#         high = q3 + 1.5 * iqr
#         df = df.loc[(df[col] >= low) & (df[col] <= high)]
#
#     return df


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
    df_nr = pd.DataFrame(imp_nr.fit_transform(cols_nr), columns=cols_nr.columns)

    return df_nr.join(df_sb, how='right')


def scaling(df, method=None):
    """Scale dataset

    method (string): can be minmax, z-score or None
    """
    df_nr = df[df.select_dtypes(include='number').columns]
    df_sb = df[df.select_dtypes(include=['category', 'bool']).columns]

    if method == 'minmax':
        transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
        mm_df_nr = pd.DataFrame(transf.transform(df_nr), columns=df_nr.columns, index=df_nr.index)
        df = mm_df_nr.join(df_sb, how='right')
    elif method == 'z-score':
        transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
        z_df_nr = pd.DataFrame(transf.transform(df_nr), columns=df_nr.columns, index=df_nr.index)
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

    _, tstX, _, tstY = train_test_split(X, y, train_size=train_size, stratify=y, random_state=RANDOM_STATE)

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

    trnX, _, trnY, _ = train_test_split(X, y, train_size=train_size, stratify=y, random_state=RANDOM_STATE)

    values = {'Train': [trnX, trnY], 'Test': [tstX, tstY]}

    return values


def applyEstimator(estimator, values, omitErrors=False):
    """Fit estimator with training set and return accuracy predicted in test set
    """
    try:
        estimator.fit(values['Train'][0], values['Train'][1])
        prdY = estimator.predict(values['Test'][0])
        return metrics.accuracy_score(prdY, values['Test'][1])
    except Exception as e:
        if not omitErrors:
            print("Oops!", e.__class__, "occurred while applying estimator.")
            print("Skipping estimator")
        return -1


def findBestParams(estimator, params, values, n_jobs=-1, verbose=1):
    """Applies grid search to determine best params for estimator
    Return best estimator and corresponding parameters
    """
    search = GridSearchCV(estimator, params, cv=3, n_jobs=n_jobs, verbose=verbose)
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

        values['Train'][0] = SelectKBest(score_func, k=k).fit_transform(values['Train'][0], values['Train'][1])

        return values

    except Exception as e:
        if not omitErrors:
            print("Oops!", e.__class__, "occurred while selecting features using {}.".format(method))
            print("Skipping feature selection")
        return values


def featureExtraction(data, target, variance):
    targetData = data.pop(target)
    mean = (data.mean(axis=0)).tolist()
    centered_data = data - mean

    pca = PCA(n_components=variance, svd_solver='full')
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


def getCombs(a, b):
    combs = itertools.product(a, b)
    if None in [comb[0] for comb in combs]:
        combs = [comb for comb in combs if comb[0] is not None]
        combs.append((None, 0))
    return combs


def findBest(data, target, correlationThresholds, varianceThresholds, featureExtractionVariances, featureSelectionMethods, numFeatures, scalingMethods,
             balancingMethods, estimators, estimatorParams, omitErrors=False, n_jobs=-1, verbose=1):
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

    for correlationThreshold in correlationThresholds:
        if correlationThreshold is not None:
            correlation_removal(data, target, correlationThreshold)
        for varianceThreshold in varianceThresholds:
            if varianceThreshold is not None:
                remove_low_variance(data, target, varianceThreshold)
            for featureExtractionVariance in featureExtractionVariances:
                if featureExtractionVariance is not None:
                    data = featureExtraction(data, target, featureExtractionVariance)
                for scalingMethod in scalingMethods:
                    data = scaling(data, scalingMethod)
                    for balancingMethod in balancingMethods:
                        values = balancing(data, target, balancingMethod)
                        for featureSelectionMethod, k in getCombs(featureSelectionMethods, numFeatures):
                            values = featureSelection(values, featureSelectionMethod, k, omitErrors)
                            for estimatorName, estimator in estimators.items():
                                bestParams = 'default'
                                bestEstimator = estimator

                                if bool(estimatorParams[estimatorName]):
                                    bestEstimator, bestParams = findBestParams(estimator, estimatorParams[estimatorName], values,
                                                                               n_jobs=n_jobs, verbose=verbose)

                                acc = applyEstimator(bestEstimator, values, omitErrors)

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
                                    'accuracy': acc
                                }

                                if acc > bestAcc:
                                    bestAcc = acc
                                    best = combination

                                allResults.append(combination)

    return best, allResults
