"""
Day 5: baseline training on NF-UNSW-NB15-v2, mirroring baseline_train.py's
NSL-KDD protocol exactly (same architecture, same training-loop params,
same FilterColumn/is_test evaluation approach) so the two datasets are a
genuine apples-to-apples comparison, not just superficially similar.
"""
import json
import sys
import time

import numpy as np
import tensorflow as tf

from framework.enumerations import EvaluationDatasetSampling
from framework.flow_transformer import FlowTransformer
from framework.flow_transformer_parameters import FlowTransformerParameters
from implementations.classification_heads import LastTokenClassificationHead
from implementations.input_encodings import NoInputEncoder
from implementations.pre_processings import StandardPreProcessing
from implementations.transformers.basic_transformers import BasicTransformer

from prepare_nf_unsw import nf_unsw_nb15_v2

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
    "NF-UNSW-NB15-v2",
    "data/NF-UNSW-NB15-v2-prepared.csv",
    nf_unsw_nb15_v2,
    evaluation_dataset_sampling=EvaluationDatasetSampling.FilterColumn,
)
print(f"[{time.time()-t0:.1f}s] Dataset loaded")

m = ft.build_model()
param_count = m.count_params()
print(f"[{time.time()-t0:.1f}s] Model built. Param count: {param_count:,}")

m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])

train_results, eval_results, final_epoch = ft.evaluate(
    m, batch_size=128, epochs=100, steps_per_epoch=128, early_stopping_patience=5
)
total_time = time.time() - t0
print(f"[{total_time:.1f}s] Training complete. Stopped at epoch {final_epoch}.")
print(eval_results)

last = eval_results.iloc[-1]
precision = last["TP"] / (last["TP"] + last["FP"]) if (last["TP"] + last["FP"]) > 0 else 0.0
recall = last["TP"] / (last["TP"] + last["FN"]) if (last["TP"] + last["FN"]) > 0 else 0.0
far = last["FP"] / (last["FP"] + last["TN"]) if (last["FP"] + last["TN"]) > 0 else 0.0

summary = {
    "dataset": "NF-UNSW-NB15-v2",
    "seed": SEED,
    "config": "Shallow encoder (2-layer, 2-head, dim 128), Last Token head, NoInputEncoder",
    "params": int(param_count),
    "final_epoch": int(final_epoch),
    "total_wall_clock_seconds": round(total_time, 1),
    "eval_split": "stratified 80/20 (FilterColumn on is_test) -- no official split exists",
    "f1": float(last["f1"]),
    "balanced_accuracy": float(last["bal_acc"]),
    "precision": float(precision),
    "detection_rate_recall": float(recall),
    "false_alarm_rate": float(far),
    "TP": int(last["TP"]), "FP": int(last["FP"]),
    "TN": int(last["TN"]), "FN": int(last["FN"]),
}

with open(f"nf_unsw_baseline_results_seed{SEED}.json", "w") as f:
    json.dump(summary, f, indent=2)

m.save_weights(f"nf_unsw_baseline_model_seed{SEED}.weights.h5")

print("\n--- SUMMARY ---")
for k, v in summary.items():
    print(f"{k}: {v}")
