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

        # add features: position of the appt in the day, and by doctor
        self.df['appt_pos_overall'] = self.df.PATIENT_ID.apply(
            self.get_appt_pos
            )
        self.df['appt_pos_doctor'] = self.df.PATIENT_ID.apply(
            lambda x: self.get_appt_pos(x, doc=True)
            )

    def drop_redundant(
            self,
            cols=[
                'VISIT_DATE',
                'PT_SCHEDULED_APPT',
                'PT_ARRIVE_TIME',
                'PT_START_TIME',
                'PT_END_TIME'
                ]
            ):
        for col in cols:
            self.df.drop(col, axis=1, inplace=True)

    def print_counts(self, col):
        c = collections.Counter(self.df[col])
        for cat, count in c.most_common():
            print_str = '{cat}\t{co}\t{per:.2f}'.format(
                cat=cat[:5],
                co=count,
                per=100*float(count)/len(self.df)
                )
            print(print_str)

    def get_appt_pos(self, pid, doc=False):
        appt_row = self.df[self.df.PATIENT_ID == pid]
        day = appt_row.date.iloc[0]
        sched_time = appt_row.sched.iloc[0]
        if doc:
            doctor = appt_row.PROVIDER_NAME.iloc[0]
            appts_that_day = self.df[
                (self.df.date == day) & (self.df.PROVIDER_NAME == doctor)
                ]
        else:
            appts_that_day = self.df[self.df.date == day]
        sched_times = appts_that_day.groupby('sched').sched.max()
        appt_pos = list(sched_times).index(sched_time)
        return appt_pos

    def make_pairplot(
            self,
            df=None,
            pair_vars=[
                'AGE',
                'delay',
                'appt_time',
                'month',
                'num_appts',
                'dayofweek'
                ],
            hue='PATIENT_CONDITION',
            file_name=None
            ):
        if df is None:
            df = self.df
        g = sns.PairGrid(
            data=df,
            vars=pair_vars,
            hue=hue
            )
        g = g.map_diag(plt.hist, edgecolor="w")
        g = g.map_offdiag(plt.scatter, edgecolor='w')
        g.add_legend(fontsize=20, markerscale=2)
        if file_name:
            g.savefig(file_name, dpi=300)
        else:
            return g

    def make_scatter(
            self,
            df=None,
            hue='PATIENT_CONDITION',
            size=4,
            xvar='AGE',
            yvar='appt_time',
            file_name=None
            ):
        if df is None:
            df = self.df
        g = sns.FacetGrid(data=df, hue=hue, size=size)
        g = g.map(plt.scatter, xvar, yvar, edgecolor='w')
        g.add_legend(fontsize=10, markerscale=2)
        if file_name:
            g.savefig('age_appt_cond.png', dpi=300)
        else:
            return g
