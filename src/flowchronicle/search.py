
from typing import Optional
import pandas as pd
import numpy as np
import copy
from itertools import product, combinations
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import pickle

import flowchronicle.dataloader as dl
from flowchronicle.model import Model
from flowchronicle.pattern import Pattern
from flowchronicle.row_pattern import RowPattern
from flowchronicle.dataloader import Dataset
from flowchronicle.attribute_value import AttributeValue, AttributeType
from flowchronicle import our_globals



def extend_row(row_pattern:Optional[RowPattern], dataset:Dataset, ignore_columns = set()) -> list[RowPattern]:
    '''
    Returns a list of extension candidates
    Also works with None, returns all possible single attribute rows
    Returns empty list if no extension possible
    '''
    pattern_dict = row_pattern.pattern if row_pattern != None else {}
    set_columns = pattern_dict.keys()
    unset_columns = set(range(dataset.shape[1])) - set_columns
    unset_columns = unset_columns - ignore_columns
    unset_columns = unset_columns - dataset.never_fix

    extensions = []
    for uc in unset_columns:
        for i in range(len(dataset.value_prob[uc])):
            new_pattern_dict = copy.deepcopy(pattern_dict)
            new_pattern_dict[uc] = AttributeValue(AttributeType.FIX, i)
            extensions.append(RowPattern(new_pattern_dict))
    return extensions

def combine_patterns(patternA:Pattern, patternB:Pattern) -> Pattern:
    inc = patternA.get_max_placeholder_id()
    patternB_copy = copy.deepcopy(patternB)
    patternB_copy.inc_placeholders_ids(inc+1)
    return Pattern(copy.deepcopy(patternA.pattern) + patternB_copy.pattern)

def get_all_ip_in_pattern(pattern:Pattern, dataset:Dataset, incl_placeholder_marker=False) -> set:
    ip_columns = dataset.ip_columns
    ip_values = set()
    for row in pattern.pattern:
        for col, val in row.pattern.items():
            if col in ip_columns:
                if val.attr_type == AttributeType.FIX:
                    ip_values.add(val.value)
                elif val.attr_type == AttributeType.USE_PLACEHOLDER or val.attr_type == AttributeType.SET_PLACEHOLDER:
                    ip_values.add(-1) # -1 to represent placeholder
    return ip_values

def incl_same_ip(patternA:Pattern, patternB:Pattern, dataset:Dataset) -> bool:
    ip_valuesA = get_all_ip_in_pattern(patternA, dataset, incl_placeholder_marker=True)
    ip_valuesB = get_all_ip_in_pattern(patternB, dataset, incl_placeholder_marker=True)
    return len(ip_valuesA.intersection(ip_valuesB)) > 0

def combine_row_only_patterns(pattern_set) -> Pattern:
    single_row_patterns = list(filter(lambda x: len(x.pattern) == 1, pattern_set))
    all_pattern_combinations = list(combinations(single_row_patterns, 2))
    new_patterns = []
    for p1,p2 in all_pattern_combinations:
        if p1 == p2:
            continue
        # find conflicting columns
        p1_cols = set(p1.pattern[0].pattern.keys())
        p2_cols = set(p2.pattern[0].pattern.keys())
        intersection = p1_cols.intersection(p2_cols)
        conflict = False
        for i in intersection:
            if p1.pattern[0].pattern[i] != p2.pattern[0].pattern[i]:
                conflict = True
                break # patterns conflict
        if conflict:
            continue
        # combine patterns
        new_patterns.append(Pattern([RowPattern({**p1.pattern[0].pattern, **p2.pattern[0].pattern})]))
    return new_patterns


def extend_with_placeholder(pattern:Pattern, dataset:Dataset) -> list[Pattern]:
    '''
    Returns a list of extension candidates
    Does not do anything if single row pattern is given
    '''
    # for each already existing set a all possible gets
    new_patterns = []
    placeholder_uid_identifier = -1
    for i, row in enumerate(pattern.pattern):
        for val in row.pattern.values():
            if val.attr_type == AttributeType.SET_PLACEHOLDER:
                placeholder_uid_identifier = max(placeholder_uid_identifier, val.value)
                for j, row_j in enumerate(pattern.pattern[i+1:], start=i+1):
                    for c in dataset.placeholder_get_columns - row_j.pattern.keys():
                        new_pattern = copy.deepcopy(pattern.pattern)
                        new_row = copy.deepcopy(row_j.pattern)
                        new_row[c] = AttributeValue(AttributeType.USE_PLACEHOLDER, val.value)
                        new_pattern[j] = RowPattern(new_row)
                        new_patterns.append(Pattern(new_pattern))
    # extend with new set and get combinations
    for i, row in enumerate(pattern.pattern):
        for c in dataset.placeholder_set_columns - row.pattern.keys():
            for j, row_j in enumerate(pattern.pattern[i+1:], start=i+1):
                for cj in dataset.placeholder_get_columns - row_j.pattern.keys():
                    new_pattern = copy.deepcopy(pattern.pattern)
                    new_row_i = copy.deepcopy(row.pattern)
                    new_row_j = copy.deepcopy(row_j.pattern)
                    placeholder_uid_identifier += 1
                    new_row_i[c] = AttributeValue(AttributeType.SET_PLACEHOLDER,placeholder_uid_identifier)
                    new_row_j[cj] = AttributeValue(AttributeType.USE_PLACEHOLDER, placeholder_uid_identifier)
                    new_pattern[i] = RowPattern(new_row_i)
                    new_pattern[j] = RowPattern(new_row_j)
                    new_patterns.append(Pattern(new_pattern))
            # extend with new row
            for cj in dataset.placeholder_get_columns - row.pattern.keys():
                new_pattern = copy.deepcopy(pattern.pattern)
                new_row_i = copy.deepcopy(row.pattern)
                placeholder_uid_identifier += 1
                new_row_i[c] = AttributeValue(AttributeType.SET_PLACEHOLDER,placeholder_uid_identifier)
                new_pattern[i] = RowPattern(new_row_i)
                new_pattern.append(RowPattern({cj:AttributeValue(AttributeType.USE_PLACEHOLDER, placeholder_uid_identifier)}))
                new_patterns.append(Pattern(new_pattern))


    return new_patterns

def extend_patterns_with_placeholder(patterns:list[Pattern], dataset:Dataset) -> list[Pattern]:
    extended_patterns =[]
    for pattern in patterns:
        extended_patterns = extended_patterns + extend_with_placeholder(pattern, dataset)
    return extended_patterns

def build_single_attribute_patterns(dataset:Dataset) -> list[Pattern]:
    '''
    Returns a list of all 1 attribute rows
    '''
    single_attribute_rows = extend_row(None, dataset)
    return list(map(lambda x: Pattern([x]), single_attribute_rows))

def build_inital_patterns(dataset:Dataset) -> list[Pattern]:
    '''
    Returns a list of all 2 attribute rows
    '''
    single_attribute_rows = extend_row(None, dataset)
    pair_attribute_rows = []
    for row in single_attribute_rows:
        pair_attribute_rows = pair_attribute_rows + extend_row(row, dataset, ignore_columns=set(range(next(iter(row.pattern.keys()))))) #ignore all larger columns

    return list(map(lambda x: Pattern([x]), pair_attribute_rows)) #TODO list best type for this?

def extend_pattern_rows(pattern:Pattern, dataset:Dataset) -> list[Pattern]:
    '''
    Extend each row in pattern with one attribute,
    returns a list of all possible extensions
    '''
    extended_patterns =[]
    for i,row in enumerate(pattern.pattern):
        for new_row in extend_row(row, dataset):
            new_pattern = copy.deepcopy(pattern.pattern)
            new_pattern[i] = new_row
            extended_patterns.append(Pattern(new_pattern))
    return extended_patterns

def extend_patterns_rows(patterns:list[Pattern], dataset:Dataset) -> list[Pattern]:
    '''
    Extend each row in pattern with one attribute,
    returns a list of all possible extensions
    '''
    extended_patterns =[]
    for pattern in patterns:
        extended_patterns = extended_patterns + extend_pattern_rows(pattern, dataset)
    return extended_patterns


def search(dataset:Dataset, load_checkpoint=0, model_name='model', load_path=None) -> Model:
    init_patterns = build_inital_patterns(dataset)
    single_attribute_patterns = build_single_attribute_patterns(dataset)
    map(lambda x: x.get_candidate_score(dataset), init_patterns) #only compute it once
    if load_path != None:
        model = Model.load_model(load_path)
    else:
        model = Model(dataset) if load_checkpoint == 0 else Model.load_model("output/{}_{}.pkl".format(model_name, load_checkpoint))

    candidate_score_cache = {}
    tested_candidates = set()
    changes = True

    DEBUG_ITERATION_LIMIT = np.inf
    search_mode = our_globals.START_SEARCH_MODE
    '''
    0 - single row only
    1 - rows and multiple rows
    '''
    iterations = load_checkpoint
    while (changes or search_mode == 0) and iterations < DEBUG_ITERATION_LIMIT:
        logging.debug("Search iteration %d" % iterations)
        iterations += 1
        if not changes:
            search_mode = 1

        changes = False

        candidates  = init_patterns\
                    + extend_patterns_rows(model.get_patterns(), dataset)\
                    + extend_patterns_with_placeholder(model.get_patterns(), dataset)\
                    + combine_row_only_patterns(model.get_patterns())
        if search_mode == 1:
            product_patterns = product(model.get_patterns() + single_attribute_patterns, repeat=2)
            product_patterns = filter(lambda x: incl_same_ip(x[0],x[1], dataset), product_patterns)
            candidates = candidates\
                        + list(map(lambda x: combine_patterns(x[0],x[1]),product_patterns))


        candidates = np.array(candidates)
        candidate_scores = list(map(lambda x: x.get_candidate_score(dataset, candidate_score_cache), candidates))



        unseccessful_candidates = 0
        while unseccessful_candidates < our_globals.MAX_UNSUCCESSFUL_CANDIDATES:
            best_candidate_id = np.argmax(candidate_scores)
            best_candidate = candidates[best_candidate_id]
            if hash(best_candidate) in tested_candidates:
                candidate_scores[best_candidate_id] = -1 # mark as useless
                continue
            else:
                logging.debug("test candidate %s" % best_candidate)
                tested_candidates.add(hash(best_candidate))
                if not model.test_add(best_candidate): # pruning is part of test_add
                    candidate_scores[best_candidate_id] = -1 # mark as useless
                    unseccessful_candidates += 1
                else:
                    logging.debug("added candidate %s" % best_candidate)
                    changes = True
                    break
    return model


if __name__ == '__main__':
    df = dl.load_socbed_bi()
    dataset = Dataset(df.copy())
    m = search(dataset)
    for p in m.get_patterns():
        print(p)
    pickle.dump(m, open('model.pkl', 'wb'))
