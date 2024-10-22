import sys
from programs.model import Model, ChunkyModel
from programs.temporal_sampling import PatternSampler
from our_train_and_generate import parrallelize_flows
import argparse
import pandas as pd

if __name__ == "__main__":
    print("start load")

    m = Model.load_model("models/our.pkl")
    print("finish load")
    c = m.cover
    cover_stats = c.get_cover_stats()
    patterns_usage = cover_stats.get_pattern_usage()

    c.fit_temporal_samplers(patterns_usage.keys())
    idx, synthetic_patterns = PatternSampler(patterns_usage).sample(sum(patterns_usage.values()),return_indices=True)
    print(len(synthetic_patterns))
    
    synthetic_df = parrallelize_flows(synthetic_patterns, c.dataset.col_name_map, c.dataset.column_value_dict, c.dataset.cont_repr, c.dataset.time_stamps, idxs=idx, cpus=45)
    synthetic_df.insert(0, 'Date first seen', synthetic_df.pop('Date first seen'))
    synthetic_df.to_csv("data/artifact_evaluation.csv", index=False)
