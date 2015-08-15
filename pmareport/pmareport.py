# -*- coding: utf-8 -*-
'''Pma report analysis.'''

import pandas as pd
import collections
import seaborn as sns
from matplotlib import pyplot as plt


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

        # drop the duplicates
        self.df = self.df.drop_duplicates()

        # change time columns to datetime formats
        self.df['date'] = pd.to_datetime(self.df.VISIT_DATE, format='%Y-%m-%d')
        self.df['sched'] = pd.to_datetime(
            self.df.PT_SCHEDULED_APPT,
            format='%H:%M'
            )
        self.df['start'] = pd.to_datetime(
            self.df.PT_START_TIME,
            format='%H:%M'
            )
        self.df['end'] = pd.to_datetime(
            self.df.PT_END_TIME,
            format='%H:%M'
            )
        self.df['arrive'] = pd.to_datetime(
            self.df.PT_ARRIVE_TIME,
            format='%H:%M'
            )

        # derive some basic time features
        self.df['lateness'] = self.df.arrive - self.df.sched
        self.df['delay'] = (self.df.start - self.df.arrive).dt.seconds/60.0
        self.df['appt_time'] = (self.df.end - self.df.start).dt.seconds/60.0
        self.df['month'] = self.df.date.dt.month
        self.df['dayofweek'] = self.df.date.dt.dayofweek

        # add feature: number of appointments that day
        num_appts = self.df.groupby('date').count().PATIENT_ID
        self.df['num_appts'] = self.df.date.apply(lambda x: num_appts[x])

    def print_counts(self, col):
        c = collections.Counter(self.df[col])
        for cat, count in c.most_common():
            print_str = '{cat}\t{co}\t{per:.2f}'.format(
                cat=cat[:5],
                co=count,
                per=100*float(count)/len(self.df)
                )
            print(print_str)

    def make_pairplot(self):
        pair_grid_vars = [
            'AGE',
            'delay',
            'appt_time',
            'month',
            'num_appts',
            'dayofweek'
            ]
        g = sns.PairGrid(
            data=self.df,
            vars=pair_grid_vars,
            hue='PATIENT_CONDITION'
            )
        g = g.map_diag(plt.hist, edgecolor="w")
        g = g.map_offdiag(plt.scatter, edgecolor='w')
        g.add_legend(fontsize=20, markerscale=2)
        g.savefig('features_by_condition.png', dpi=300)
