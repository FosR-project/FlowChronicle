from __future__ import annotations
from flowchronicle import window
from flowchronicle.dataloader import Dataset
from flowchronicle.row_pattern import RowPattern
from flowchronicle.temporal_sampling import TemporalSampler
from flowchronicle import mdl_util
from flowchronicle import bayesian_network
from flowchronicle.attribute_value import AttributeValue, AttributeType


from typing import Optional
import warnings
import copy
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Pattern: # sequence of RowPatterns
    def __init__(self, pattern:list[RowPattern]):
        assert len(pattern) > 0
        self.pattern = pattern
        self.bn = None
        self.temporal_sampler = None
        self.__window_cache = None
        self.__candidate_score = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(list(map(str, self.pattern))) + str(hash(self))

    def __hash__(self) -> int:
        return hash(tuple(self.pattern))

    def get_real_value_repr(self, dataset:Dataset) -> str:
        str_rep = ""
        for row in map(lambda x: x.get_real_value_repr(dataset), self.pattern):
            str_rep += row + "\n"
        return str_rep

    def __complete_window(self, start_position:int, set_vars:dict, dataset:Dataset) -> Optional[window.Window]:
        ids = [start_position]
        all_false_mask = pd.Series(False, index=dataset.flow_features.index)

        for row in self.pattern[1:]:
            last_row_match = row.next_row_match(dataset,ids[-1], set_vars)
            if last_row_match == None:
                return None

            all_false_mask[last_row_match] = True
            new_set_vars = row.init_vars(all_false_mask, dataset)[0]
            all_false_mask[last_row_match] = False
            set_vars = {**set_vars, **new_set_vars}
            ids.append(last_row_match)
        return window.Window(ids, self)

    def find_windows(self, dataset:Dataset) -> list[window.Window]:
        if self.__window_cache != None:
            return self.__window_cache
        logging.debug("Find windows for pattern %s" % self)
        windows = []
        firstrow = self.pattern[0]
        start_positions = firstrow.matching_rows(dataset)
        var_sets = firstrow.init_vars(start_positions,dataset)
        for i, set_vars in zip(dataset.flow_features.index[start_positions],var_sets):
            opt_window = self.__complete_window(i,set_vars, dataset)
            if opt_window != None:
                windows.append(opt_window)
        self.__window_cache = windows
        return windows

    def get_candidate_score(self, dataset:Dataset, candidate_score_cache) -> float: # higher is better, -1 us useless
        if self.__candidate_score != None:
            return self.__candidate_score
        if self.__hash__() in candidate_score_cache:
            self.__candidate_score = candidate_score_cache[self.__hash__()]
            return self.__candidate_score

        conflict_map = np.full(dataset.shape[0], True)
        added_windows = 0
        for w in sorted(self.find_windows(dataset), key=lambda x:x.score(), reverse=True):
            if np.all(conflict_map[w.ids]):
                conflict_map[w.ids] = False
                added_windows += 1
        self.__candidate_score = added_windows * sum([x.get_n_coverd_fields() for x in self.pattern])
        candidate_score_cache[self.__hash__()] = self.__candidate_score #TODO do not add single row length two patterns
        return self.__candidate_score


    def fit_bayesian_network(self, dataset:Dataset, windows:list[window.Window]):
        self.bn = bayesian_network.BayesianNetwork()
        self.bn.learn(dataset, windows)

    def build_pattern_with_const_values(self) -> Optional[Pattern]:
        assert self.bn != None
        if len(self.bn.const_vars.items()) == 0:
            return None
        else:
            new_pattern = copy.deepcopy(self.pattern)
            for var, value in self.bn.const_vars.items():
                colum_row = var.split("-r")
                column = int(colum_row[0])
                row = int(colum_row[1])
                if column in new_pattern[row].pattern.keys():
                    logging.debug("column %s in row %s already in pattern" % (column, row))
                    if new_pattern[row].pattern[column].attr_type == AttributeType.SET_PLACEHOLDER:
                        set_uuid_key = new_pattern[row].pattern[column].value
                        new_pattern[row].pattern[column] = AttributeValue(AttributeType.FIX, int(value))
                        # 1 find all USE_PLACEHOLDER in pattern and replace with new FIX value
                        for r in new_pattern:
                            for c, v in r.pattern.items():
                                if v.attr_type == AttributeType.USE_PLACEHOLDER and v.value == set_uuid_key:
                                    r.pattern[c] = AttributeValue(AttributeType.FIX, int(value))
                    else:
                        print(self)
                        print(new_pattern)
                        print(self.bn.const_vars)
                        raise Exception("BUG BUG")
                else:
                    new_pattern[row].pattern[column] = AttributeValue(AttributeType.FIX, int(value))
            return Pattern(new_pattern)

    def get_temporal_sampler(self, time_delays, optimize=False):
        return TemporalSampler(time_delays, opti=optimize) ##To optimize

    def sample_timestamps(self, init_time, end_time):
        init_timestamp = self.temporal_sampler[0].generate_first_timestamp()
        while not init_time <= init_timestamp < end_time:
            init_timestamp = self.temporal_sampler[0].generate_first_timestamp()
        timestamps=[init_timestamp]
        #if len(self.pattern)>1:
        for i, ts in enumerate(self.temporal_sampler[1:]):
            dt = ts.sample(1)
            if timestamps[i] == [[None]] or dt > end_time - timestamps[i]:
                timestamps.append([[None]])
            else :
                timestamps.append(timestamps[i]+dt)
        return np.asarray(timestamps).flatten()

    def get_delay_cost(self, i:int, delay:int) -> float:
        assert i < len(self.pattern)
        return mdl_util.length_natural(delay+1) # +1 since timepoints can be the same

    def get_encoding_cost(self, dataset:Dataset, allow_const_in_bn = False) -> float:
        cost = 0.0
        cost += mdl_util.length_natural(len(self.pattern))
        cost += self.pattern[0].get_encoding_cost(True,dataset,0)
        set_placeholders = self.pattern[0].get_number_of_set_vars()
        for p in self.pattern[1:]:
            cost += p.get_encoding_cost(False,dataset,set_placeholders)
            set_placeholders += p.get_number_of_set_vars()
        cost += self.bn.get_cost(encode_constant_variables=allow_const_in_bn)
        return cost

    def get_max_placeholder_id(self) -> int:
        ids = [p.get_max_placeholder_id() for p in self.pattern]
        return max(ids) if len(ids) > 0 else -1

    def inc_placeholders_ids(self, inc:int):
        for p in self.pattern:
            p.inc_placeholders_ids(inc)


class EmptyPattern(Pattern):
    def __init__(self):
        self.pattern = []
        self.bn = None

    def find_windows(self, dataset: Dataset) -> list[window.Window]:
        to_empty_window = np.vectorize(lambda x: window.EmptyWindow(x, self))
        return to_empty_window(dataset.time_stamps.index)

    def get_encoding_cost(self, dataset:Dataset) -> float:
        return 0.0

    def get_delay_cost(self, i: int, delay: int) -> float:
        raise Exception("Empty pattern has no delay cost")
