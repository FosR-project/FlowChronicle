import warnings
warnings.simplefilter('once', UserWarning)

from typing import Optional, Dict
import bisect
import math
import scipy

import numpy as np

from flowchronicle.attribute_value import AttributeValue, AttributeType
from flowchronicle.dataloader import Dataset
from flowchronicle import mdl_util
import collections

class RowPattern: # describing a subset of the attributes of a row / flow
    def __init__(self, pattern:dict[int,AttributeValue]):
        assert len(pattern) > 0 #at least one attribute needs to be used in anyway
        assert all([isinstance(x,int) for x in pattern.keys()]) # no set placeholders allowed
        self.pattern = pattern
        self.type_counts = collections.Counter([x.attr_type for x in pattern.values()])


    def __hash__(self) -> int:
        return hash(tuple(self.pattern.items()))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(sorted(list(self.pattern.items()), key=lambda x: x[0]))

    def get_real_value_repr(self, dataset:Dataset) -> str:
        return str({dataset.col_name_map[x]:y.get_real_value_repr(x,dataset) for x,y in self.pattern.items()})

    def get_n_coverd_fields(self) -> int:
        return len(self.pattern) - self.type_counts[AttributeType.SET_PLACEHOLDER]

    #untested method since patterns do not incl. vars yet!
    def init_vars(self, matching_rows, dataset): #-> list[Dict]:
        columns = []
        val_vector = []
        for col, val in self.pattern.items():
            if val.attr_type == AttributeType.SET_PLACEHOLDER:
                columns.append(col)
                val_vector.append(val.value)
        to_dict = lambda x: dict(zip(val_vector,x))
        return dataset.flow_features[matching_rows].apply(to_dict, axis=1).to_list()



    def matching_rows(self, dataset): #  -> np.ndarray[bool]:
        d = dataset.flow_features
        columns = []
        val_vector = []
        for col, val in self.pattern.items():
            assert val.attr_type != AttributeType.USE_PLACEHOLDER # method meant for first row matches
            if val.attr_type != AttributeType.SET_PLACEHOLDER:
                columns.append(col)
                val_vector.append(val.value)
        mask = np.all(dataset.flow_features[columns] == val_vector, axis=1)
        return  mask #list(dataset.flow_features.index[mask])

    def next_row_match(self, dataset:Dataset, row:int, set_vars:dict) -> Optional[int]:
        columns = []
        val_vector = []
        for col, val in self.pattern.items():
            if val.attr_type != AttributeType.SET_PLACEHOLDER:
                columns.append(col)
                if val.attr_type == AttributeType.FIX:
                    val_vector.append(val.value)
                elif val.attr_type == AttributeType.USE_PLACEHOLDER:
                    val_vector.append(set_vars[val.value])
                else:
                    raise Exception("BUG BUG!")
        mask = np.all(dataset.flow_features[columns] == val_vector, axis=1)
        ids = list(dataset.flow_features.index[mask])
        next_row = bisect.bisect(ids, row)
        if next_row == len(ids):
            return None
        return ids[next_row]

    def get_number_of_set_vars(self) -> int:
        return self.type_counts[AttributeType.SET_PLACEHOLDER]

    def value_encoding_cost(self, dataset:Dataset, col:int, value:int) -> float:
        return math.log2(len(dataset.value_prob[col]))

    def get_encoding_cost(self, first:bool, dataset:Dataset, n_set_placeholders:int) -> float:
        cost = 0.0
        # fix values
        columns = dataset.shape[1]
        cost += math.log2(columns)
        cost += mdl_util.log_choose(columns, self.type_counts[AttributeType.FIX])
        for col, val in self.pattern.items():
            if val.attr_type == AttributeType.FIX:
                cost += self.value_encoding_cost(dataset, col, val.value)

        if not first and n_set_placeholders > 0 and (columns - self.type_counts[AttributeType.FIX] - self.type_counts[AttributeType.SET_PLACEHOLDER]) > 0:
            cost += math.log2(columns - self.type_counts[AttributeType.FIX] - self.type_counts[AttributeType.SET_PLACEHOLDER]) # how many of the prev are used in this row
            # only one usage per row allowed
            cost += mdl_util.log_choose(columns - self.type_counts[AttributeType.FIX], self.type_counts[AttributeType.USE_PLACEHOLDER]) #which columns
            cost += self.type_counts[AttributeType.USE_PLACEHOLDER] * math.log2(n_set_placeholders) # which set placeholders

        # set placeholder values
        remaining_columns = columns - self.type_counts[AttributeType.FIX] - self.type_counts[AttributeType.USE_PLACEHOLDER]
        cost += math.log2(remaining_columns) if remaining_columns > 0 else 0 # which columns
        cost += mdl_util.log_choose(columns - self.type_counts[AttributeType.FIX] - self.type_counts[AttributeType.USE_PLACEHOLDER], self.type_counts[AttributeType.SET_PLACEHOLDER]) # which columns

        return cost

    def get_max_placeholder_id(self) -> int:
        ids = [p.value for p in self.pattern.values() if p.attr_type == AttributeType.SET_PLACEHOLDER]
        return max(ids) if len(ids) > 0 else -1

    def inc_placeholders_ids(self, inc:int):
        for p in self.pattern.values():
            if p.attr_type == AttributeType.SET_PLACEHOLDER or p.attr_type == AttributeType.USE_PLACEHOLDER:
                p.value += inc
