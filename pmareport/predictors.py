# -*- coding: utf-8 -*-
'''PMA analytic tools'''

import pandas as pd
from sklearn import tree
from sklearn import cross_validation
import numpy as np


def read_data(fp='../data/pmadata.csv'):
    '''Read clinic data from a csv into a pandas dataframe.'''
    return pd.read_csv(fp)


def percent_within(y_true, y_pred, thresh=5):
    '''
    Calculate the percentage of predictions are within
    `thresh` of the true value.
    '''
    return np.sum(np.abs(y_true - y_pred) < thresh)/float(len(y_true))*100


class DurationPredictor(object):
    '''
    A model to predict the duration of an appointment.

    For example,

    '''
    def __init__(self, df, feat_cols, response_col):
        self.df = df
        self.feat_cols = feat_cols
        self.response_col = response_col

    def make_int(self, col):
        '''
        Encode categorical variables as ints
        for input into the decision tree.
        '''
        categories = list(set(self.df[col]))
        int_func = lambda x: categories.index(x)
        self.df[col+'i'] = self.df[col].apply(int_func)
        self.feat_cols.remove(col)
        self.feat_cols.append(col+'i')

    def train_test(self, test_size=0.1):
        '''
        Split the data into train and test sets.
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
        '''
        self.model = tree.DecisionTreeRegressor(max_depth=max_depth)

    def cv_evalution(self, n_folds=10, thresh=5):
        '''
        Evaluate the model on a cross valdation split
        of the training data with `n_folds` nmber of folds.
        The metric is the percent of predictions within `thresh`
        of the true value.

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
