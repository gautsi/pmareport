# -*- coding: utf-8 -*-
'''Pma report analysis.'''

import pandas as pd


def read_data(fp='../data/pmadata.csv'):
    return pd.read_csv(fp)


class Clinic(object):
    '''
    A collection of functions for the pma report analysis.
    '''

    def __init__(self, df=None, fp='../data/pmadata.csv'):
        if df is None:
            self.df = read_data(fp)
        else:
            self.df = df
        self.df = self.df.drop_duplicates()
