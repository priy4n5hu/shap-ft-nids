"""
Shared infrastructure for Day 3 (SHAP) and Day 4 (faithfulness/stability).

The tricky part: FlowTransformer classifies a flow using a *window* of the
preceding `window_size` flows (self-attention over the window), with the
LastToken head reading only the final position's representation. So
"explaining flow i" means: hold the flow's real preceding context fixed,
and let SHAP perturb only flow i's own feature values (the last position
in the window) -- this mirrors standard per-instance feature attribution
while respecting the model's actual windowed input contract.

This module rebuilds FlowTransformer's internal feature_columns_map logic
(numeric feature -> single column; one-hot categorical feature -> list of
columns) exactly as flow_transformer.py's evaluate() does internally, so
the windowed arrays we hand to model.predict() are byte-for-byte the same
shape/order the model was trained on.
"""
import numpy as np
import pandas as pd

from framework.dataset_specification import NamedDatasetSpecifications
from framework.enumerations import EvaluationDatasetSampling
from framework.flow_transformer import FlowTransformer
from framework.flow_transformer_parameters import FlowTransformerParameters
from framework.enumerations import CategoricalFormat
from implementations.classification_heads import LastTokenClassificationHead
from implementations.input_encodings import NoInputEncoder
from implementations.pre_processings import StandardPreProcessing
from implementations.transformers.basic_transformers import BasicTransformer


def build_flow_transformer():
    """Rebuilds the exact architecture used in baseline_train.py (untrained)."""
    pre_processing = StandardPreProcessing(n_categorical_levels=32)
    ft = FlowTransformer(
        pre_processing=pre_processing,
        input_encoding=NoInputEncoder(),
        sequential_model=BasicTransformer(2, 128, n_heads=2),
        classification_head=LastTokenClassificationHead(),
        params=FlowTransformerParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1),
    )
    return ft


def load_trained(seed: int, data_path: str = "data/NSL-KDD.csv",
                  dataset_name: str = "NSL-KDD", dataset_spec=None,
                  weights_path: str = None):
    """Loads data + rebuilds model + restores trained weights for a given seed.

    Defaults preserve the original NSL-KDD behavior used by Day 3/4 scripts.
    For a different dataset (e.g. NF-UNSW-NB15-v2), pass dataset_name,
    dataset_spec (a framework.dataset_specification.DatasetSpecification),
    and weights_path explicitly.
    """
    if dataset_spec is None:
        dataset_spec = NamedDatasetSpecifications.nsl_kdd
    if weights_path is None:
        weights_path = f"baseline_model_seed{seed}.weights.h5"

    ft = build_flow_transformer()
    ft.load_dataset(
        dataset_name, data_path, dataset_spec,
        evaluation_dataset_sampling=EvaluationDatasetSampling.FilterColumn,
    )
    m = ft.build_model()
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])
    m.load_weights(weights_path)
    return ft, m


def build_feature_columns_map(ft):
    """Exact replica of FlowTransformer's internal samplewise_to_featurewise mapping:
    numeric feature -> its own column name; one-hot categorical -> list of its
    dummy columns. Also returns the column-index version (into ft.X.columns)
    for fast numpy slicing.
    """
    spec = ft.model_input_spec
    feature_columns_map = {}
    for feature in spec.feature_names:
        if feature in spec.numeric_feature_names or spec.categorical_format == CategoricalFormat.Integers:
            feature_columns_map[feature] = [feature]
        else:
            feature_columns_map[feature] = [c for c in ft.X.columns if str(c).startswith(feature)]

    all_columns = list(ft.X.columns)
    feature_col_idx_map = {
        feature: [all_columns.index(c) for c in cols]
        for feature, cols in feature_columns_map.items()
    }
    return feature_columns_map, feature_col_idx_map, all_columns


class WindowedInstanceExplainer:
    """For one target flow (by row index i1 into ft.X), fixes the real preceding
    window context and exposes a predict_fn(Z) -> P(attack) that SHAP can call,
    where Z is (n_samples, n_flat_features) in ft.X's column order and only the
    TARGET position (the flow itself) varies across rows of Z.
    """

    def __init__(self, ft, model, feature_col_idx_map, all_columns):
        self.ft = ft
        self.model = model
        self.feature_col_idx_map = feature_col_idx_map
        self.all_columns = all_columns
        self.feature_names = ft.model_input_spec.feature_names
        self.window_size = ft.parameters.window_size
        self.numeric_feature_names = list(ft.model_input_spec.numeric_feature_names)

    def numeric_column_indices(self) -> np.ndarray:
        """Flat (0..87) column indices that are numeric features -- the only
        columns it's meaningful to add small Gaussian noise to for a stability
        test (categorical one-hot columns must stay a valid one-hot pattern)."""
        idx = []
        for f in self.numeric_feature_names:
            idx.extend(self.feature_col_idx_map[f])
        return np.array(sorted(idx))

    def background_mean_vector(self, background: np.ndarray) -> np.ndarray:
        """Mean of the background sample -- the 'baseline/unknown' value used
        when masking a feature out for faithfulness testing. For one-hot
        categorical groups this is a soft (non-one-hot) average vector, which
        is the standard, simple convention for occlusion-based faithfulness
        tests (as opposed to KernelSHAP's own sampling-based masking)."""
        return background.mean(axis=0)

    def flat_vector_for_index(self, i1: int) -> np.ndarray:
        """The real, observed flat feature vector for flow i1 (what we explain)."""
        return self.ft.X.iloc[i1][self.all_columns].values.astype("float32")

    def context_for_index(self, i1: int) -> dict:
        """Real preceding window_size-1 rows, split per feature -> (window_size-1, k) arrays."""
        context_df = self.ft.X.iloc[(i1 - self.window_size) + 1: i1]
        assert len(context_df) == self.window_size - 1, "index too close to start of dataframe"
        context = {}
        for feature in self.feature_names:
            cols = [self.all_columns[j] for j in self.feature_col_idx_map[feature]]
            context[feature] = context_df[cols].values.astype("float32")  # (window_size-1, k)
        return context

    def make_predict_fn(self, i1: int):
        context = self.context_for_index(i1)
        model = self.model
        feature_names = self.feature_names
        feature_col_idx_map = self.feature_col_idx_map

        def predict_fn(Z: np.ndarray) -> np.ndarray:
            n = Z.shape[0]
            inputs = []
            for feature in feature_names:
                idx = feature_col_idx_map[feature]
                target_vals = Z[:, idx].astype("float32")            # (n, k)
                ctx = context[feature]                                 # (window_size-1, k)
                ctx_tiled = np.broadcast_to(ctx, (n,) + ctx.shape).copy()  # (n, ws-1, k)
                target_vals = target_vals.reshape(n, 1, -1)             # (n, 1, k)
                full_window = np.concatenate([ctx_tiled, target_vals], axis=1)  # (n, ws, k)
                inputs.append(full_window)
            preds = model.predict(inputs, verbose=0, batch_size=256)
            return preds.reshape(-1)

        return predict_fn
