from __future__ import annotations
from enum import Enum
from flowchronicle.dataloader import Dataset

class AttributeType(Enum):
    FIX = 1
    USE_PLACEHOLDER = 2
    SET_PLACEHOLDER = 3

class AttributeValue:
    # FIX - value: corresponds to value in data
    # USE_VAR - value: int starting with 0 mapping to set var
    # SET_VAR - value: int starting with 0 mapping to use var
    def __init__(self, attr_type:AttributeType, value:int):
        assert isinstance(attr_type, AttributeType)
        assert isinstance(value, int)
        self.attr_type = attr_type
        self.value = value

    def __hash__(self) -> int:
        return hash((self.attr_type, self.value))

    def __repr__(self):
        return str(self.attr_type) + ":" + str(self.value)

    def get_real_value_repr(self, attr, dataset:Dataset) -> str:
        if self.attr_type.value == AttributeType.FIX.value: # get False if not directly comparing on value
            col = dataset.col_name_map[attr]
            if self.value in dataset.column_value_dict[col].keys():
                return dataset.column_value_dict[col][self.value]
            else: return None
        elif self.attr_type.value == AttributeType.USE_PLACEHOLDER.value:
            return "USE_PLACEHOLDER" + str(self.value)
        elif self.attr_type.value == AttributeType.SET_PLACEHOLDER.value:
            return "SET_PLACEHOLDER" + str(self.value)
        else:
            raise Exception("BUG BUG!")
