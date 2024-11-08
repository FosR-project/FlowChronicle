from flowchronicle.dataloader import Dataset
from flowchronicle import window
from flowchronicle.attribute_value import AttributeValue, AttributeType
from flowchronicle import mdl_util

import copy

import itertools
import logging
import math
import numpy as np
import pandas as pd
import pgmpy
import pgmpy.models
import pgmpy.estimators
import pgmpy.sampling
import pgmpy.factors.discrete

class NotAHackEstimator(pgmpy.estimators.BayesianEstimator):
    """ Nothing to see here, move along (a small hack so "fit" doesn’t normalize the tables)
    """
    def estimate_cpd(self, node, prior_type="BDeu", pseudo_counts=[], equivalent_sample_size=5, weighted=False): # ignored
        parents = sorted(self.model.get_parents(node))
        parents_cardinalities = [len(self.state_names[parent]) for parent in parents]

        return pgmpy.factors.discrete.TabularCPD(node, len(self.state_names[node]), np.array(self.state_counts(node)), evidence=parents, evidence_card=parents_cardinalities, state_names={var: self.state_names[var] for var in itertools.chain([node], parents)})

class BayesianNetwork:
    def __init__(self, max_indegree:int = 3):
        self.bn = None
        self.dataset = None
        self.const_vars = {} # dict of constant variables mapping to their value
        # self.col_name_map = None # dict between internal BN name like "r1c2" to feature name like "Proto". Multiple BN variables can map to the same feature.
        self.col_number_map = {} # dict between internal BN name and feature number, so "rxcy" maps to "y"
        self.row_number_map = {} # dict between internal BN name and row number, so "rxcy" maps to "x"
        self.pattern_length = None
        self.domains = None
        self.learned = False
        self.train_data = None
        self.max_indegree = max_indegree

    def _construct_one_window(self, dataset:Dataset, w: window.Window) -> dict:
        # transform one window into one dictionary
        cov_rows, cov_cols = w.get_covered_fields()
        covered = [(row, cov_cols[i]) for i,row in enumerate(cov_rows)]
        d = {} # output dictionary
        update_map = False
        if len(self.col_number_map) == 0:
            self.pattern_length = len(w.ids)
            update_map = True
            # self.col_name_map = {}
            self.col_number_map = {}
            self.row_number_map = {}
        for i, row in enumerate(w.ids):
            for col in dataset.flow_features.columns:
                if (row,col) not in covered:
                    varname = str(col).replace(" ", "")+"-r"+str(i)
                    d[varname] = dataset.flow_features[col][row]
                    if update_map:
                        # self.col_name_map[varname] = dataset.col_name_map[col]
                        self.col_number_map[varname] = col
                        self.row_number_map[varname] = i
        return d

    def _construct_training_set(self, dataset:Dataset, windows: list[window.Window]) -> pd.DataFrame:
        # transform a set of windows to a dataframe and update the list of constant variables
        df = pd.DataFrame.from_records([self._construct_one_window(dataset, w) for w in windows])
        return df

    def _drop_constant_variables(self, df:pd.DataFrame) -> pd.DataFrame:
        assert self.learned
        return df.drop(self.const_vars, axis=1)

    def learn(self, dataset:Dataset, windows: list[window.Window]):
        """ Learn the structure of the BN and its CPTs
        """
        self.train_window_set = copy.deepcopy(windows) # keep a copy of the windows used for training
        # BIC score is equivalent-ish to MDL
        self.dataset = dataset
        self.learned = True
        df = self._construct_training_set(dataset, windows)
        self.const_vars = {v: df[v].iloc[0] for v in df.columns[df.nunique()==1]} # set of constant columns
        df = self._drop_constant_variables(df)
        self.train_data = df
        if not df.empty:
            self._update_domains()
            scoring_method = pgmpy.estimators.BicScore(data=df)
            est = pgmpy.estimators.HillClimbSearch(data=df)
            self.bn = pgmpy.models.BayesianNetwork(est.estimate(scoring_method=scoring_method, max_indegree=self.max_indegree, show_progress=False))
            # self.bn.fit(data=df,estimator=pgmpy.estimators.BayesianEstimator,prior_type="K2")
            self.bn.fit(data=df,estimator=NotAHackEstimator) # force the exact count for computing cost

    def fit(self):
        """ Fit the CPT. Should be called before sampling.
        """
        if not self.train_data.empty:
            self.bn.fit(data=self.train_data,estimator=pgmpy.estimators.BayesianEstimator,prior_type="K2")

    def _get_cardinality(self, v) -> int:
        """ Get the cardinality of the variable in the whole dataset
        """
        varname = self.col_number_map[v]
        return self.dataset.flow_features.nunique()[varname]

    def _update_domains(self):
        """ Update the domains of the variables (in the whole dataset)
        """
        dataset_domains = {}
        for v in self.dataset.flow_features.columns:
            dataset_domains[v] = self.dataset.flow_features[v].unique() # get all uniques values for the columns from the dataset
        self.domains = {}
        for k,v in dataset_domains.items(): # for each column in the dataset set
            for k2,v2 in self.col_number_map.items(): # check all variables in the BN that refer to that column
                if v2 == k:
                    l = self.domains.get(k2,[]) + list(v) # add to the domain already computed
                    self.domains[k2] = list(set(l)) # remove duplicated

    def show_model(self):
        """ Show a graphical representation of the Bayesian network structure
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        assert self.learned
        print("Independencies", self.bn.get_independencies())
        print("CPTs")
        for cpt in self.bn.get_cpds():
            print(repr(cpt))
        nx.draw_circular(self.bn, with_labels=True, arrowsize=30, node_size=800, alpha=0.3, font_weight="bold")
        plt.show()

    def get_cost(self, encode_constant_variables: bool = False, verbose: bool = False) -> float:
        """ Get the cost of the model
        """
        assert self.learned

        # structure description
        cost_par_nb = 0
        cost_which_par = 0
        cost_const = 0

        if encode_constant_variables == False:
            assert len(self.const_vars) == 0

        if encode_constant_variables:
            cost_const = len(self.dataset.flow_features.columns) # what variables are constant (1 bit per variable)
            for v in self.const_vars.keys(): # domain description of constant variables
                cost_const += math.log2(self._get_cardinality(v)) # indicate which one is the constant value

        if self.bn is not None:
            m = len(self.bn.nodes) # number of nodes
            for v in self.bn.nodes:
                k = self.bn.in_degree(v) # number of parents
                cost_par_nb += math.log2(self.max_indegree) # encode the number of parents
                # cost_par_nb += mdl_util.length_natural(k + 1) # k+1 because k=0 is possible
                # cost_which_par += k * math.log2(m - 1) # which parents
                cost_which_par += math.log2(math.comb(m - 1, k)) # which parents, other version
                # it’s m-1 because a node cannot be its own parent)

        if verbose:
            print("BN cost breakdown:", cost_par_nb, "bits (how many parents per node),", cost_which_par, "bits (which parents)", cost_const, "bits (value of constant variables). Total:", cost_par_nb + cost_which_par + cost_const, "bits")

        return cost_par_nb + cost_which_par + cost_const

    def get_data_cost(self) -> float:
        """ Compute the cost of the encoding of the data passed to the "learn" function
        Compute the cost of the last windows set defined with "get_sample_cost" OR the full set of windows if "get_sample_cost" was not called after "learn"
        """
        assert self.learned

        if self.bn is None: # only constant variables
            return 0

        total_cost = 0
        epsilon = 0.5
        for v in self.bn.nodes:
            cpt = self.bn.get_cpds(v)
            list_of_counts = np.array(cpt.get_values()).T
            for counts in list_of_counts:
                total_cost += np.sum(np.log2(np.arange(0,sum(counts)) + cpt.cardinality[0] * epsilon)) # denominator
                total_cost -= sum([np.sum(np.log2(np.arange(0, i) + epsilon)) for i in counts]) # numerator
        return total_cost

    def get_sample_cost(self, window_set: list[window.Window]) -> float:
        """ Compute the cost of encoding the window_set, BN must be already learned, raises error if window set is not compatible with the BN
        In practice we assume window_set to be a subset of the learned windows
        """
        assert self.learned
        if self.bn is None:
            return 0
        df = self._construct_training_set(self.dataset, window_set)
        if not df.empty:
            df = self._drop_constant_variables(df)
        if not df.empty:
            self.bn.fit(data=df,estimator=NotAHackEstimator) # force the exact count for computing cost
            return self.get_data_cost()
        return 0 # only constant variables

    def sample(self, size:int = 1) -> list[pd.DataFrame]:
        """ Generate "size" samples
        """
        assert self.learned

        output = [pd.DataFrame(index=range(self.pattern_length), columns=self.dataset.flow_features.columns) for k in range(size)]
        if self.bn is not None:
            self._sample(output)
        return output

    def _sample(self, l:list[pd.DataFrame]) -> list[pd.DataFrame]:
        """ DataFrame in parameters are modified in place
        """
        inference = pgmpy.sampling.BayesianModelSampling(self.bn)
        vectors = inference.forward_sample(len(l), show_progress=True) # generate vectors
        for index, row in vectors.iterrows(): # for each generated vector
            for c in self.col_number_map: # fill variables in BN
                if c not in row.index: #We do not want to check for the value that are constant variables
                    continue
                l[index][self.col_number_map[c]][self.row_number_map[c]] = row[c]
            for k,v in self.const_vars.items(): # fill constant variables
                l[index][self.col_number_map[k]][self.row_number_map[k]] = v
