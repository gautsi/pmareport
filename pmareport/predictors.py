# -*- coding: utf-8 -*-
'''
The model used to predict appointment duration is a decision tree.
The model is evaluated by the precentage of predicted times that are within
a threshold (5 minutes by default) of the actual duration.

The class `DurationPredictor` splits the data into testing and training, builds
the model (using scikit-learn's implementation of decision tree)
and evaluates the model both on a cross validation split of the training set
and on the test set.

`DurationPredictor` also includes functionality to
turn non-integer categorical features into ints, which scikit-learn's decision
tree implementation requires.
'''

import pandas as pd
from sklearn import tree
from sklearn import cross_validation
import numpy as np


def read_data(fp='../data/pmadata.csv'):
    '''
    Read clinic data from a csv into a pandas dataframe.

    :param str fp: the file path of the csv file
    '''
    return pd.read_csv(fp)


def percent_within(y_true, y_pred, thresh=5):
    '''
    Calculate the percentage of predictions are within
    `thresh` of the true value.

    :param array-like y_true: the true values
    :param array-like y_pred: the predicted values
    :param float thresh: the threshold for a close prediction

    :returns:
        the percent of predictions within the treshold from the true value
    :rtype: float
    '''
    return np.sum(np.abs(y_true - y_pred) < thresh)/float(len(y_true))*100


class DurationPredictor(object):
    '''
    A model to predict the duration of an appointment.

    For example, let's make a dataframe with random data in columns `feat1`
    and `response`.

    >>> df = pd.DataFrame(np.random.randn(30,2), columns=['feat1', 'response'])

    We add a column `feat2` with categorical values ('a' or 'b').

    >>> df['feat2'] = np.random.choice(['a', 'b'], 30)

    Let's make a `DurationPredictor` object from our example dataframe.

    >>> dec_pred = DurationPredictor(
    ...     df=df,
    ...     feat_cols=['feat1', 'feat2'],
    ...     response_col='response'
    ...     )

    To turn `feat2` into a column of ints
    (which scikit-learn's decision tree implementation requires),
    we use `make_int`.

    >>> dec_pred.make_int(col='feat2')

    We split our data set into train and test with 10% left out to test.

    >>> dec_pred.train_test(test_size=0.1)

    Now let's make the model, a decision tree of maximum depth 3,
    and get its average score on a 10-fold cross validation split.
    The score is the percentage of predictions within 5 minutes
    of the acutal value.

    >>> dec_pred.make_model(max_depth=3)
    >>> cv_score = dec_pred.cv_evalution(thresh=5)
    >>> cv_score >= 0 and cv_score <= 100
    True

    Fit the model on the full training set and evaluate it on the test set.

    >>> test_score = dec_pred.fit()
    >>> test_score >= 0 and test_score <= 100
    True

    :param dataframe df: the data
    :param list feat_cols: a list of the names of the feature columns
    :param str response_col: the name of the response column

    '''
    def __init__(self, df, feat_cols, response_col):
        self.df = df
        self.feat_cols = feat_cols
        self.response_col = response_col
        self.int_funcs = {}

    def make_int(self, col):
        '''
        Encode categorical variables of type other than int
        as ints for input into the decision tree.

        :param str col: the name of the column with categorical values
        '''

        categories = list(set(self.df[col]))
        int_func = lambda x: categories.index(x)
        self.df[col+'i'] = self.df[col].apply(int_func)
        self.feat_cols.remove(col)
        self.feat_cols.append(col+'i')
        self.int_funcs[col] = int_func

    def train_test(self, test_size=0.1):
        '''
        Split the data into train and test sets.

        :param float test_size: the percentage of rows to leave out as test
        '''

        self.train, self.test = cross_validation.train_test_split(
            self.df,
            test_size=test_size
            )
        self.Xtrain = self.train[self.feat_cols]
        self.ytrain = self.train[self.response_col]
        self.Xtest = self.test[self.feat_cols]
        self.ytest = self.test[self.response_col]

    def make_model(self, max_depth=3):
        '''
        Make the model, a decision tree with maximum depth `max_depth`.

        :param max_depth: the maximum depth of the decision tree
        '''
        self.model = tree.DecisionTreeRegressor(max_depth=max_depth)

    def cv_evalution(self, n_folds=10, thresh=5):
        '''
        Evaluate the model on a cross valdation split
        of the training data with `n_folds` nmber of folds.
        The metric is the percent of predictions within `thresh`
        of the true value.

        :param int n_folds: the number of folds for the cross validation
        :param float thresh:
            the threshold for considering a prediction close to the true value

        :returns: the average of metric values over the folds
        :rtype: float
        '''
        cv = cross_validation.KFold(len(self.train), n_folds=n_folds)
        score_list = []
        for train, test in cv:
            cvXtrain = self.Xtrain.iloc[train]
            cvXtest = self.Xtrain.iloc[test]
            cvytrain = self.ytrain.iloc[train]
            cvytest = self.ytrain.iloc[test]
            self.model.fit(cvXtrain, cvytrain)
            pred = self.model.predict(cvXtest)
            score = percent_within(y_true=cvytest, y_pred=pred, thresh=5)
            score_list.append(score)
        return np.mean(score_list)

    def fit(self, thresh=5):
        '''
        Fit the model on the training set and evaluate it
        on the test set. The metric is the percent of
        predictions within `thresh` of the true value.

        :param float thresh:
            the threshold for considering a prediction close to the true value

        :returns: the score of the model on the test set
        :rtype: float
        '''
        self.model.fit(self.Xtrain, self.ytrain)
        predictions = self.model.predict(self.Xtest)
        score = percent_within(
            y_true=self.ytest,
            y_pred=predictions,
            thresh=thresh
            )
        return score
