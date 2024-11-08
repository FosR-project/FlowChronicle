from __future__ import annotations
from flowchronicle.pattern import Pattern, EmptyPattern
from flowchronicle import bayesian_network
from flowchronicle.cover import Cover
from flowchronicle import mdl_util
from flowchronicle.attribute_value import AttributeType, AttributeValue

import numpy as np

import pickle
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class Model:
    def __init__(self, dataset, **kwargs) -> None:
        self.__base_bn = bayesian_network.BayesianNetwork()
        all_empty_windows = EmptyPattern().find_windows(dataset)
        self.__base_bn.learn(dataset, all_empty_windows)
        self.pattern_set = []
        self.cover = Cover(self.pattern_set, self.__base_bn, dataset)
        self.cover.greedy_search()
        self.encode_const_var_in_bn = kwargs.get('encode_const_var_in_bn', False)


    def get_model_length(self) -> float:
        '''
        length of bayesian nework
        \LN(number of patterns)
        sum over patterns:
        '''
        cost = 0.0
        cost += mdl_util.length_natural(len(self.pattern_set)+1)
        for p in self.pattern_set:
            cost += p.get_encoding_cost(self.cover.dataset, allow_const_in_bn=self.encode_const_var_in_bn)
        cost += self.__base_bn.get_cost() #TODO do want to include this- constant cost does not change anything?
        return cost

    def get_patterns(self) -> list[Pattern]: #TODO assess return type
        return self.pattern_set

    def test_add(self, pattern:Pattern) -> bool:
        '''
        Test if pattern does improve MDL score, add to model if it does
        '''
        tmp_cover = Cover(self.pattern_set + [pattern], self.__base_bn, self.cover.dataset)
        tmp_cover.greedy_search()
        pattern.fit_bayesian_network(self.cover.dataset, [w for w in tmp_cover.get_used_windows() if w != None and w.pattern == pattern])
        while len(pattern.bn.const_vars) != 0:
            logging.debug("Extend pattern with const values from %s" % pattern)
            logging.debug("const vars %s" % pattern.bn.const_vars)
            pattern = pattern.build_pattern_with_const_values()
            logging.debug("to pattern %s" % pattern)
            # repeat steps with const values incl. in pattern -- not the most efficient way
            tmp_cover = Cover(self.pattern_set + [pattern], self.__base_bn, self.cover.dataset)
            tmp_cover.greedy_search()
            pattern.fit_bayesian_network(self.cover.dataset, [w for w in tmp_cover.get_used_windows() if w != None and w.pattern == pattern])
        assert len(pattern.bn.const_vars) == 0
        if tmp_cover.compute_data_length() + pattern.get_encoding_cost(self.cover.dataset) < self.cover.compute_data_length():
            old_cover_usage = self.cover.get_pattern_usage()
            new_cover_usage = tmp_cover.get_pattern_usage()

            prune_candidates = []
            for p,usage in old_cover_usage.items():
                diff = usage - new_cover_usage[p]
                if diff > 0:
                    prune_candidates.append((p,diff))

            self.pattern_set.append(pattern)
            self.cover = tmp_cover
            self.prune_patterns(prune_candidates)
            return True
        else:
            return False

    def prune_patterns(self, candidates): #only test patterns where the usage changed
        '''
        Remove patterns that do not improve MDL score
        '''
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        for p,_ in sorted_candidates:
            if self.cover.compute_data_length([p]) + p.get_encoding_cost(self.cover.dataset) > self.cover.compute_data_length():
                self.cover.remove_pattern(p)
                self.pattern_set.remove(p)



    def set_pattern_set(self, pattern_set:list[Pattern]):
        self.pattern_set = pattern_set
        self.cover = Cover(self.pattern_set, self.__base_bn, self.cover.dataset)
        self.cover.greedy_search()
        self.cover.fit_bayesian_networks(self.pattern_set)

    def get_base_bn(self):
        return self.__base_bn

    def save_model(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    @staticmethod
    def load_model(filename) -> Model:
        return pickle.load(open(filename, 'rb'))

class ChunkyModel(Model):

    def __init__(self, dataset, chunks ,**kwargs) -> None:
        self.__base_bn = bayesian_network.BayesianNetwork()
        all_empty_windows = EmptyPattern().find_windows(dataset)
        self.__base_bn.learn(dataset, all_empty_windows)
        self.encode_const_var_in_bn = kwargs.get('encode_const_var_in_bn', False)
        self.pattern_set = []
        self.cover = Cover(self.pattern_set, self.__base_bn, dataset)


        m0 = chunks[0]

        for p in m0.pattern_set:
            if p.bn.bn is not None: #Some pattern do not have BN, because they have an empty windows list. We should not include those in the final model
                for row in p.pattern:
                    for k in row.pattern.keys():
                        if row.pattern[k].attr_type == AttributeType.FIX:
                            column = m0.cover.dataset.col_name_map[k]
                            real_value = m0.cover.dataset.column_value_dict[column][row.pattern[k].value]
                            row.pattern[k] = AttributeValue(AttributeType.FIX, self.cover.dataset.column_value_dict[column].inverse[real_value])



        self.pattern_set.extend(m0.pattern_set)
        self.cover.pattern_set = self.pattern_set
        self.cover.cover_map = m0.cover.cover_map

        offset = m0.cover.cover_map.shape[0]

        for m in chunks[1:]:
            print("Merging next chunk")
            for p in m.pattern_set:
                if p.bn.bn is not None: #Some pattern do not have BN, because they have an empty windows list. We should not include those in the final model
                    for row in p.pattern:
                        for k in row.pattern.keys():
                            if row.pattern[k].attr_type == AttributeType.FIX:
                                column = m.cover.dataset.col_name_map[k]
                                real_value = m.cover.dataset.column_value_dict[column][row.pattern[k].value]
                                row.pattern[k] = AttributeValue(AttributeType.FIX, self.cover.dataset.column_value_dict[column].inverse[real_value])

            self.pattern_set.extend(m.pattern_set)
            self.cover.pattern_set = self.pattern_set
            seen = set()
            for w in m.cover.cover_map.flatten()[m.cover.cover_map.flatten() != None]:
                if id(w) in seen:
                    continue
                else:
                    seen.add(id(w))
                    w.ids = [i + offset for i in w.ids]

            self.cover.cover_map = np.vstack((self.cover.cover_map, m.cover.cover_map))
            offset += m.cover.cover_map.shape[0]




