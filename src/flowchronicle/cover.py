from __future__ import annotations
from queue import PriorityQueue
from collections import Counter
import numpy as np
import collections
import warnings
warnings.simplefilter('once', UserWarning)
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
import math

from flowchronicle import mdl_util
from flowchronicle.dataloader import Dataset
from flowchronicle.pattern import Pattern, EmptyPattern
from flowchronicle.window import Window, EmptyWindow
from flowchronicle.bayesian_network import BayesianNetwork
import random

class Cover:
    def __init__(self, pattern_set:list[Pattern], base_baysian_network:BayesianNetwork, dataset:Dataset):
        self.pattern_set = pattern_set
        self.dataset = dataset
        self.cover_map = None
        self.cover_stats = None
        self.base_baysian_network = base_baysian_network
        self.independent_timepoint_cost = self.__compute_independent_timepoint_cost()
        self.empty_pattern_added = False
        self.__cache_data_length_cost = None
        self.__uid = random.randint(1, 922337203685477580)

    def __compute_independent_timepoint_cost(self):
        min_timestamp =self.dataset.time_stamps.iloc[0]
        max_timestamp = self.dataset.time_stamps.iloc[-1]
        return np.log2(max_timestamp - min_timestamp)

    def __get_used_windows(self):
        if self.cover_map is None:
            raise Exception("Cover not yet computed")
        else:
            cnt = collections.Counter(self.cover_map.ravel()) # np.unique does not work
            return cnt.keys()

    def get_used_windows(self): #added for visualization i.e. show pattern usage in data
        if self.cover_map is None:
            raise Exception("Cover not yet computed")
        else:
            return self.__get_used_windows()

    def set_empty_cover(self):
        self.cover_map = np.empty(self.dataset.shape, dtype=Window)
        self.__cache_data_length_cost = None

    def greedy_search(self):
        assert not self.empty_pattern_added
        # its here so that it only shows up once!
        # warnings.warn("Using np.all to find next window - probably inefficient")
        # warnings.warn("Window.score: Using simple number of fields covered as score")

        self.__cache_data_length_cost = None

        logging.debug("Starting window search")
        window_list = []
        # queue = PriorityQueue()
        for p in self.pattern_set:
            window_list += p.find_windows(self.dataset)

        window_scores = np.array(list(map(lambda w: w.score(), window_list)))
        window_list = np.array(window_list)[(-window_scores).argsort()]



        self.cover_map = np.empty(self.dataset.shape, dtype=Window)
        self.covered_rows = np.zeros(self.dataset.shape[0], dtype=bool)

        logging.debug("Starting covering, Cover %d" % self.__uid)
        iter_counter = 0
        for w in window_list:
            iter_counter += 1
            if iter_counter % 1000 == 0:
                logging.debug("Iteration %d" % iter_counter)
            # _, w = queue.get()
            p = w.pattern

            rows, cols = w.get_covered_fields()
            u_rows = np.unique(rows)

            # if any of the rows are already covered, skip and add alternative to queue.
            # we go rows to ensure that a row is at most covered by only one window, this way we can learn small Bayesian networks for the non covered rows
            # essentially the non covered fields are covered by the Bayesian network.
            blocked = np.any(self.covered_rows[u_rows])
            if not blocked:
                self.cover_map[rows, cols] = w
                self.covered_rows[u_rows] = True
            # else:
            #     # Potential speed up, do not add next window if fist symboled is already covered
            #     next_w = p.get_next_window(self.dataset, w)
            #     if next_w != None:
            #         queue.put((-next_w.score(self.dataset), next_w))


        logging.debug("Finished covering, Cover %d" % self.__uid)

    def fit_bayesian_networks(self, pattern_set:set[Pattern]):
        self.__cache_data_length_cost = None
        windows = self.__get_used_windows()
        for p in pattern_set:
            used_windows = [w for w in windows if w != None and w.pattern == p]
            p.fit_bayesian_network(self.dataset, used_windows)

    def get_cover_map(self):
        if self.cover_map is None:
            raise Exception("Cover not yet computed")
        else:
            return self.cover_map

    def compute_data_length(self, excepted_patterns = []) -> float:
        if self.__cache_data_length_cost is not None and excepted_patterns == []:
            return self.__cache_data_length_cost

        logging.debug("Computing data length cost for cover %d" % self.__uid)
        windows = self.__get_used_windows()
        mask_covered_rows = np.zeros(self.dataset.shape[0], dtype=bool)

        windows_dict = {}
        data_cost:float = 0.0
        for w in windows:
            # ignore Nones, corresponds to uncovered fields
            # collect all window ids
            if w != None and w.pattern not in excepted_patterns:
                window_cost:float = 0.0
                covered_rows = self.dataset.time_stamps.iloc[w.ids]
                mask_covered_rows[w.ids] = True
                # delay cost i.e. time point cost
                window_cost += self.independent_timepoint_cost
                # compute delta between covered rows
                for i, delay in enumerate(np.diff(covered_rows)):
                    window_cost += w.pattern.get_delay_cost(i, delay)
                data_cost += window_cost

                w_list = windows_dict.get(id(w.pattern), [])
                w_list.append(w)
                windows_dict[id(w.pattern)] = w_list

        for _,l in windows_dict.items():
            #compute the cost of the uncovered fields by calling the bayesian network of the pattern
            try:
                data_cost += mdl_util.length_natural(len(l))
                data_cost += l[0].pattern.bn.get_sample_cost(l) # already in bits
            except AttributeError as e:
                logging.debug("No Bayesian network for pattern %s" % l[0].pattern)
                raise e


        windows = [EmptyWindow.from_no_pattern(i) for i in np.where(mask_covered_rows == False)[0]]
        data_cost += self.base_baysian_network.get_sample_cost(windows)
        data_cost += len(windows) * self.independent_timepoint_cost

        # for i in np.where(mask_covered_rows == False)[0]:
            #compute the cost of the uncovered fields by calling the base bayesian network
            # data_cost -= math.log2(self.base_baysian_network.get_probability(self.dataset, EmptyWindow.from_no_pattern(i)))
            # data_cost += self.independent_timepoint_cost

        if excepted_patterns == []:
            self.__cache_data_length_cost = data_cost
        return data_cost

    def remove_pattern(self,pattern):
        self.pattern_set.remove(pattern)
        pattern_windows = np.vectorize(lambda w:w != None and w.pattern == pattern)(self.cover_map)
        print(pattern_windows)
        self.cover_map[pattern_windows] = None
        self.__cache_data_length_cost = None

    def fit_temporal_samplers(self, pattern_set:set[Pattern]):
        time_delays = self.cover_stats.get_time_delays()
        init_timestamps = self.cover_stats.get_pattern_start_times()
        for p in pattern_set:
            #p.temporal_sampler = [p.get_temporal_sampler(np.asarray(init_timestamps[p]), optimize=True)]
            p.temporal_sampler = [p.get_temporal_sampler(np.asarray(init_timestamps[p]), optimize=False)] #We do not optimize the |KDE, instead we use the default para;eters
            p.temporal_sampler += [None]*len(time_delays[p])
            for idx, tds in time_delays[p].items():
                p.temporal_sampler[idx+1] = p.get_temporal_sampler(np.asarray(tds))

    def get_pattern_usage(self) -> dict[Pattern, int]:
        used_windows = self.__get_used_windows()
        used_patterns = [w.pattern for w in used_windows if w != None]
        patterns_usage = collections.Counter(used_patterns)
        return patterns_usage

    def get_cover_stats(self)-> CoverStatistics:
        if self.cover_map is None:
            raise Exception("Cover not yet computed")
        else:
            empty_pattern = EmptyPattern()
            self.pattern_set.append(empty_pattern)
            self.empty_pattern_added = True

            cover_stats = CoverStatistics()
            uncovered_rows = np.all(self.cover_map == None, axis=1)
            empty_windows = empty_pattern.find_windows(self.dataset)
            used_empty_windows = empty_windows[uncovered_rows]
            used_windows = self.__get_used_windows()
            used_windows = list(used_windows) + list(used_empty_windows)
            # fit Bayesian network for empty window
            empty_pattern.bn = BayesianNetwork()
            empty_pattern.bn.learn(self.dataset, used_empty_windows)


            used_patterns = [w.pattern for w in used_windows if w != None]
            patterns_usage = collections.Counter(used_patterns) #incl. empty pattern, return of get_pattern_usage does not include empty pattern
            cover_stats.set_pattern_usage(patterns_usage)


            time_delays = {}
            start_times = {}
            for p in patterns_usage.keys():
                time_delays[p] = {}
                start_times[p] = []
                for i in range(len(p.pattern)-1):
                    time_delays[p][i] = []
            for w in used_windows:
                if w != None:
                    covered_rows = self.dataset.time_stamps.iloc[w.ids]
                    start_times[w.pattern].append(covered_rows.iloc[0])
                    for i, delay in enumerate(np.diff(covered_rows)):
                        time_delays[w.pattern][i].append(delay)
            cover_stats.set_time_delays(time_delays)
            cover_stats.set_pattern_start_times(start_times)
            self.cover_stats = cover_stats
            return cover_stats


class CoverStatistics:
    '''
    Statistics about cover used to generate new synthetic data!
    Using setter and getter for easy extensibility of the class
    '''
    def __init__(self) -> None:
        pass
        self.__pattern_usage:dict[Pattern, int] = None
        self.__uncovered_flows:int = None
        self.__time_delays:dict[Pattern, dict[int, list[int]]] = None #usage time_delays[pattern][gap_nr] gives list of time delays
        self.__pattern_start_times:dict[Pattern, list[int]] = None #usage pattern_start_times[pattern] gives list of start times

    def set_pattern_usage(self, pattern_usage:dict[Pattern, int]):
        self.__pattern_usage = pattern_usage
    def get_pattern_usage(self, init_timestamp=None, end_timestamp=None) -> dict[Pattern, int]:
        if self.__pattern_usage is None:
            raise Exception("Pattern usage not yet set!")
        if self.__pattern_start_times is None:
            raise Exception("Pattern start times not yet set!")
        if init_timestamp is None and end_timestamp is None:
            return self.__pattern_usage
        patt_usage = dict()
        for patt in self.__pattern_usage.keys():
            count = 0
            for time in self.__pattern_start_times[patt]:
                if init_timestamp <= time < end_timestamp:
                    count+=1
            patt_usage[patt] = count
        return patt_usage

    def set_uncovered_flows(self, uncovered_flows:int):
        self.__uncovered_flows = uncovered_flows
    def get_uncovered_flows(self) -> int:
        if self.__uncovered_flows is None:
            raise Exception("Uncovered flows not yet set!")
        return self.__uncovered_flows

    def set_time_delays(self, time_delays:dict[Pattern, dict[int, list[int]]]):
        self.__time_delays = time_delays
    def get_time_delays(self) -> dict[Pattern, dict[int, list[int]]]:
        if self.__time_delays is None:
            raise Exception("Time delays not yet set!")
        return self.__time_delays

    def set_pattern_start_times(self, pattern_start_times:dict[Pattern, list[int]]):
        self.__pattern_start_times = pattern_start_times
    def get_pattern_start_times(self) -> dict[Pattern, list[int]]:
        if self.__pattern_start_times is None:
            raise Exception("Pattern start times not yet set!")
        return self.__pattern_start_times
