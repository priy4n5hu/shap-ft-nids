import numpy as np
from shap_utils import load_trained, build_feature_columns_map, WindowedInstanceExplainer

ft, m = load_trained(seed=42)
feature_columns_map, feature_col_idx_map, all_columns = build_feature_columns_map(ft)
print("Num original features:", len(feature_columns_map))
print("Num flat (one-hot expanded) columns:", len(all_columns))

expl = WindowedInstanceExplainer(ft, m, feature_col_idx_map, all_columns)

indices_test = np.argwhere(~ft.training_mask).reshape(-1)
print("First few test indices:", indices_test[:5], "... total:", len(indices_test))

i1 = indices_test[100]  # arbitrary, well past the window edge
print(f"\nExplaining flow at index {i1}, true label = {ft.y[i1]}")

flat_vec = expl.flat_vector_for_index(i1)
print("Flat vector shape:", flat_vec.shape)

predict_fn = expl.make_predict_fn(i1)

# Sanity check 1: predict_fn on the REAL (unperturbed) vector should match
# the model's real prediction for this flow via the normal windowed path.
pred_via_wrapper = predict_fn(flat_vec.reshape(1, -1))
print("Prediction via SHAP wrapper (real values):", pred_via_wrapper)

# Sanity check 2: compare against ft's own windowing for the same index.
real_window = ft.X.iloc[(i1 - 8) + 1: i1 + 1]
featurewise = []
for feature in ft.model_input_spec.feature_names:
    cols = [all_columns[j] for j in feature_col_idx_map[feature]]
    featurewise.append(real_window[cols].values.astype("float32").reshape(1, 8, -1))
pred_via_direct = m.predict(featurewise, verbose=0).reshape(-1)
print("Prediction via direct FlowTransformer windowing:", pred_via_direct)

print("\nMatch:", np.allclose(pred_via_wrapper, pred_via_direct, atol=1e-5))
