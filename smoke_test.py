"""
Day-1 smoke test: verifies the full FlowTransformer pipeline (preprocessing ->
input encoding -> shallow transformer -> LastToken head -> train/eval) runs
end-to-end against the real NSL-KDD.csv we prepared, using the framework's
real API (not a guessed one). Kept tiny (few steps, 1 epoch, small sample)
so it finishes fast on CPU -- this is NOT the baseline run, just a pipeline
sanity check per Day 1 of the timeline.
"""
import os
import time
import pandas as pd

from framework.dataset_specification import NamedDatasetSpecifications
from framework.enumerations import EvaluationDatasetSampling
from framework.flow_transformer import FlowTransformer
from framework.flow_transformer_parameters import FlowTransformerParameters
from implementations.classification_heads import LastTokenClassificationHead
from implementations.input_encodings import NoInputEncoder
from implementations.pre_processings import StandardPreProcessing
from implementations.transformers.basic_transformers import BasicTransformer

t0 = time.time()

pre_processing = StandardPreProcessing(n_categorical_levels=32)

# Shallow 2-layer encoder + Last Token head, matching the blueprint's
# "Shallow encoder, Last Token" baseline configuration.
ft = FlowTransformer(
    pre_processing=pre_processing,
    input_encoding=NoInputEncoder(),
    sequential_model=BasicTransformer(2, 128, n_heads=2),
    classification_head=LastTokenClassificationHead(),
    params=FlowTransformerParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1)
)

ft.load_dataset(
    "NSL-KDD",
    os.path.join("data", "NSL-KDD.csv"),
    NamedDatasetSpecifications.nsl_kdd,
    evaluation_dataset_sampling=EvaluationDatasetSampling.RandomRows,
    evaluation_percent=0.05,
)
print(f"[{time.time()-t0:.1f}s] Dataset loaded")

m = ft.build_model()
m.summary()
print(f"[{time.time()-t0:.1f}s] Model built. Param count: {m.count_params():,}")

m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])
print(f"[{time.time()-t0:.1f}s] Model compiled")

# Tiny run: just enough to prove train + eval loop executes without error.
train_results, eval_results, final_epoch = ft.evaluate(
    m, batch_size=128, epochs=1, steps_per_epoch=10, early_stopping_patience=1
)
print(f"[{time.time()-t0:.1f}s] Smoke test evaluate() completed")
print("\n--- eval_results ---")
print(eval_results)
print(f"\nTotal smoke test time: {time.time()-t0:.1f}s")
