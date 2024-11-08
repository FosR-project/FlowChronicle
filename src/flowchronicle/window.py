from __future__ import annotations
from typing import Tuple
import numpy as np

import warnings
warnings.simplefilter('once', UserWarning)


from flowchronicle import pattern
from flowchronicle.dataloader import Dataset
from flowchronicle.attribute_value import AttributeValue, AttributeType

class Window:
    def __init__(self, ids:list[int], p:pattern.Pattern):
        self.ids = ids
        self.pattern = p

    def __gt__(self, other:Window):
        return self.score() > other.score()

    def __lt__(self, other:Window):
        return self.score() < other.score()

    def __ge__(self, other:Window):
        return self.score() >= other.score()

    def __le__(self, other:Window):
        return self.score() <= other.score()


    def score(self) -> float:
        epsilon = 0.01 #when same number of windows covered prefer the ones with fewer gaps
        gaps = np.sum(np.diff(self.ids) - 1)
        return len(self.get_covered_fields()[0]) - epsilon * gaps

    def get_covered_fields(self) -> Tuple[list[int],list[int]]:
        rows = []
        cols = []
        for i, row in enumerate(self.ids):
                for col, attr_val in self.pattern.pattern[i].pattern.items():
                    if attr_val.attr_type != AttributeType.SET_PLACEHOLDER: # set var does not cover a field
                        rows.append(row)
                        cols.append(col)
        return rows, cols

class EmptyWindow(Window):
    def __init__(self, id:int, pattern:pattern.EmptyPattern):
        super().__init__([id],pattern)

    @classmethod
    def from_no_pattern(cls,id:int):
        return cls(id, pattern.EmptyPattern())

    def get_covered_fields(self) -> Tuple[list[int], list[int]]:
        return [], []
