"""
Day 2: real baseline training run (not a smoke test). Shallow 2-layer
transformer + Last Token head, trained on the official KDDTrain+ split and
evaluated on the official KDDTest+ split (via EvaluationDatasetSampling.
FilterColumn against the 'is_test' column) -- a proper generalization test,
since KDDTest+ deliberately includes attack types absent from KDDTrain+.

Uses FlowTransformer's own default training-loop parameters (epochs=100,
steps_per_epoch=128, early_stopping_patience=5), i.e. NOT the tiny smoke-test
values -- this is meant to produce a real, reportable number.

NOTE ON REPRODUCIBILITY: FlowTransformer seeds its internal RandomState from
OS entropy unless one is explicitly supplied, and Keras weight init is also
unseeded by default -- so re-running this script (even unmodified, on the
same machine) will NOT reproduce bit-identical numbers to a previous run.
A fixed SEED is set below for exact repeatability of a single run; for the
paper's final reported numbers, consider running 2-3 different seeds and
reporting mean +/- std rather than a single run, given how cheap each run
is (~3 minutes on CPU).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from framework.dataset_specification import NamedDatasetSpecifications
from framework.enumerations import EvaluationDatasetSampling
from framework.flow_transformer import FlowTransformer
from framework.flow_transformer_parameters import FlowTransformerParameters
from implementations.classification_heads import LastTokenClassificationHead
from implementations.input_encodings import NoInputEncoder
from implementations.pre_processings import StandardPreProcessing
from implementations.transformers.basic_transformers import BasicTransformer

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
rs = np.random.RandomState(SEED)

t0 = time.time()

pre_processing = StandardPreProcessing(n_categorical_levels=32)

ft = FlowTransformer(
    pre_processing=pre_processing,
    input_encoding=NoInputEncoder(),
    sequential_model=BasicTransformer(2, 128, n_heads=2),
    classification_head=LastTokenClassificationHead(),
    params=FlowTransformerParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1),
    rs=rs,
)

ft.load_dataset(
    "NSL-KDD",
    os.path.join("data", "NSL-KDD.csv"),
    NamedDatasetSpecifications.nsl_kdd,
    evaluation_dataset_sampling=EvaluationDatasetSampling.FilterColumn,
)
print(f"[{time.time()-t0:.1f}s] Dataset loaded (official KDDTrain+/KDDTest+ split)")

m = ft.build_model()
param_count = m.count_params()
print(f"[{time.time()-t0:.1f}s] Model built. Param count: {param_count:,}")

m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])

train_results, eval_results, final_epoch = ft.evaluate(
    m, batch_size=128, epochs=100, steps_per_epoch=128, early_stopping_patience=5
)
total_time = time.time() - t0
print(f"[{total_time:.1f}s] Training complete. Stopped at epoch {final_epoch}.")
print("\n--- eval_results ---")
print(eval_results)

last = eval_results.iloc[-1]
precision = last["TP"] / (last["TP"] + last["FP"]) if (last["TP"] + last["FP"]) > 0 else 0.0
recall = last["TP"] / (last["TP"] + last["FN"]) if (last["TP"] + last["FN"]) > 0 else 0.0
detection_rate = recall
far = last["FP"] / (last["FP"] + last["TN"]) if (last["FP"] + last["TN"]) > 0 else 0.0

summary = {
    "seed": SEED,
    "config": "Shallow encoder (2-layer, 2-head, dim 128), Last Token head, NoInputEncoder",
    "params": int(param_count),
    "final_epoch": int(final_epoch),
    "total_wall_clock_seconds": round(total_time, 1),
    "eval_split": "official KDDTest+ (FilterColumn on is_test)",
    "f1": float(last["f1"]),
    "balanced_accuracy": float(last["bal_acc"]),
    "precision": float(precision),
    "detection_rate_recall": float(detection_rate),
    "false_alarm_rate": float(far),
    "TP": int(last["TP"]), "FP": int(last["FP"]),
    "TN": int(last["TN"]), "FN": int(last["FN"]),
}

with open(f"baseline_results_seed{SEED}.json", "w") as f:
    json.dump(summary, f, indent=2)

eval_results.to_csv(f"baseline_eval_results_full_seed{SEED}.csv", index=False)

m.save_weights(f"baseline_model_seed{SEED}.weights.h5")
print(f"Wrote baseline_model_seed{SEED}.weights.h5")

print("\n--- SUMMARY (for Table I) ---")
for k, v in summary.items():
    print(f"{k}: {v}")
print("\nWrote baseline_results.json and baseline_eval_results_full.csv")
