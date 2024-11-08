#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:27:34 2023

@author: aschoen
"""
import copy
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from flowchronicle.dataloader import reconstruct_bytes
from flowchronicle.attribute_value import AttributeType, AttributeValue
from scipy.optimize import minimize

class MarginalSampler():
    def __init__(self, data, method='kde', opti=False, **kwargs):
        self.method = method
        self.data=data
        if len(data.shape) == 1:
            self.data=self.data.reshape(-1,1)
        if self.method == 'kde':
            if opti:
                optimal_bw = self.optimize()
                self.estimator = KernelDensity(bandwidth=optimal_bw)
            else:
                self.estimator = KernelDensity(**kwargs)
        else :
            raise Exception("Other method than KDE are not implemented yet")

    def optimize(self, bw_min=0.05, init=1, bw_max=None, method='L-BFGS-B'):
        res = minimize(self.neg_log_likelihood_kde, x0=init, method=method, bounds=[(bw_min, bw_max)])
        return res.x[0]

    def train(self):
        self.estimator = self.estimator.fit(self.data)
        return self.estimator

    def sample(self, n):
        return self.estimator.sample(n)

    # Define the negative log-likelihood function
    def neg_log_likelihood_kde(self, bandwidth):
        #TODO Find a way to minimize overfitting
        kde = KernelDensity(bandwidth=bandwidth[0], kernel='gaussian')
        kde.fit(self.data)
        # We use the score_samples method which returns the log of the probability density
        logprob = kde.score_samples(self.data)
        # The negative log likelihood is the negative sum of log probabilities
        return -logprob.sum()

    def grid_search_KDE(self):
    #TODO see if grid search find a maxima of likelihood
        raise Exception("Not implemented yet")

class PatternSampler():
    def __init__(self, patterns_distrib):
        patt_list = []
        for patt in patterns_distrib:
            if patt.bn.bn is not None:
                patt.bn.fit()
                patt_list.append(patt)
        self.pattern_dic = dict(zip(list(range(len(patt_list))), patt_list))
        self.pattern_distrib = dict((k, patterns_distrib[k]) for k in patt_list)
        total = sum(self.pattern_distrib.values()) #Normalization
        for key in self.pattern_distrib:
            self.pattern_distrib[key] = self.pattern_distrib[key] / total
    def sample(self, n, return_indices=False):
        idxs = np.random.choice(list(self.pattern_dic.keys()), size=n, p=list(self.pattern_distrib.values()))
        if return_indices:
            return idxs, [self.pattern_dic[i] for i in idxs]
        else:
            return None, [self.pattern_dic[i] for i in idxs]

class TemporalSampler():
    def __init__(self, timestamps, **kwargs):
        #TODO
        self.time_sampler = MarginalSampler(timestamps, **kwargs).train()

    def generate_first_timestamp(self):
        return self.time_sampler.sample(1)

    def sample(self,n):
        return self.time_sampler.sample(n)

class FlowSampler():
    def __init__(self, pattern, col_name_map, column_value_dict, cont_repr, timestamps):
        self.pattern = copy.deepcopy(pattern)
        self.col_name_map = col_name_map
        self.column_value_dict = column_value_dict
        self.cont_repr = cont_repr
        self.timestamps = timestamps
        self.col_list = list(self.col_name_map.values())
        for i, flow in enumerate(self.pattern.pattern):
            c_idx = list(flow.pattern.keys())
            #columns = [self.dataset.col_name_map[i] for i in c_idx]
            columns = [self.col_name_map[c] for c in c_idx]
            v = list(flow.pattern.values())
            #v = [j.get_real_value_repr(c_idx[k],self.dataset) for k, j in enumerate(v)]
            v = [j.value if str(j).split(":")[0] == str(AttributeType.FIX) else j for k, j in enumerate(v)]
            flow.pattern = dict(zip(columns,v))
    def get_flows(self):
        self.timestamps = self.pattern.sample_timestamps(min(self.timestamps), max(self.timestamps))
        res = pd.DataFrame(columns=self.col_list)
        for fl in self.pattern.pattern:
            flows=[list(fl.pattern.values())]
            columns = list(fl.pattern.keys())
            df = pd.DataFrame(np.asarray(flows),columns=columns)
            res=pd.concat([res,df],axis=0, ignore_index=True)
        if len(res) == 0: #If the pattern is "uncovered" then create a line of NaN
            res = pd.DataFrame(np.nan, index=[0],columns=self.col_list)
        #self.pattern.bn.fit()
        synthetic_df = self.pattern.bn.sample()[0]
        synthetic_df.columns = res.columns
        #synthetic_df.dropna(inplace=True, axis=1)

        set_placeholders = [val for val in res.stack().unique() if str(val).split(":")[0] == str(AttributeType.SET_PLACEHOLDER)]
        use_placeholders = [val for val in res.stack().unique() if str(val).split(":")[0] == str(AttributeType.USE_PLACEHOLDER)]
        placeholders_values = [synthetic_df.iloc[i, j] for value in set_placeholders for i, j in zip(*np.where(res == value))]
        placeholders = dict(zip(set_placeholders, placeholders_values))
        use_placeholders_idx = [res.stack()[res.stack() == item].index[0] for item in set_placeholders]
        res.replace(placeholders, inplace=True)

        result = synthetic_df.fillna(res)
        result.replace(self.column_value_dict,inplace=True)
        d = dict(zip(use_placeholders, [result.loc[idx] for idx in use_placeholders_idx]))
        result.replace(d, inplace=True)
        #result['Date first seen'] = pd.Series(self.timestamps)/self.dataset.cont_repr.get_time_precision()
        result['Date first seen'] = pd.Series(self.timestamps)/self.cont_repr.get_time_precision()
        result['Date first seen'] = pd.to_timedelta(result['Date first seen'], 's')+self.cont_repr.get_first_flow_time()
        for c in ['In Byte', 'Out Byte', 'In Packet', 'Out Packet', 'Duration']:
        #for c in ['In Byte', 'Out Byte']:
          #  result.loc[:,c+'1']=result[c].apply(lambda x: self.dataset.cont_repr.get_cutpoints()[c][int(x)+1])
          #  result.loc[:,c] = result[c].apply(lambda x: self.dataset.cont_repr.get_cutpoints()[c][int(x)])
          #  result.loc[:,c] = np.random.uniform(result[c], result[c+'1'])
          #  result.drop(c+'1', axis=1,inplace=True)
            result[c] = reconstruct_bytes(result[c], self.cont_repr.get_cutpoints())
            if c != "Duration":
                result[c] = result[c].round().astype(int)
        #result.loc[:,temp.columns]=temp
        result.dropna(inplace=True)
        return result

if __name__ == '__main__':
    print('TODO')
