"""
iSES 2026 Paper 358 -- edge-cost benchmark (answers Reviewers 2 and 6).

Measures, for ONE trained instance, on the machine it runs on:
  1. model size: trainable parameters and weights-file size;
  2. memory: process RSS after loading the data, after loading the model, and peak;
  3. inference latency for ONE window (batch 1, the deployment case): median / p95 over many calls;
  4. batch throughput (windows per second at batch 256);
  5. cost of explaining ONE flow: one KernelSHAP explanation (500 samples, 40-flow background,
     same wrapper as the paper) and the full per-flow evaluation the paper runs
     (1 explanation + 1 sigma=0 re-explanation + 5 perturbed re-explanations).

Safety: it first re-checks the column order by reproducing the instance's stored test-set
confusion matrix (--expect), exactly like revision_eval_fo.py, and aborts on a mismatch.

Run from the project folder that holds revision_eval_fo.py, shap_utils.py, framework/, data/ and cache/
(venv active). Close other heavy programs first; keep the laptop plugged in.

  python edge_cost_benchmark.py --dataset nsl    --weights baseline_model_seed47.weights.h5        --expect baseline_results_seed47.json        --label lap47
  python edge_cost_benchmark.py --dataset nfunsw --weights nf_unsw_baseline_model_seed46.weights.h5 --expect nf_unsw_baseline_results_seed46.json --label lap46
  (optional, single CPU thread, closer to a small edge device)
  python edge_cost_benchmark.py --dataset nsl    --weights baseline_model_seed47.weights.h5        --expect baseline_results_seed47.json        --label lap47 --threads 1

Each run takes about 5-15 minutes. It writes edge_cost_<dataset>_<label>[_t1].json and prints a summary.
Send back the printed SUMMARY blocks (or the .json files).
"""
import argparse
import json
import os
import platform
import resource
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except Exception:
            return float("nan")
    return float("nan")


def peak_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if sys.platform != "darwin" else r / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nsl", "nfunsw"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--expect", required=True, help="stored baseline results JSON for this weights file")
    ap.add_argument("--feature-order", default=None, help="only for development-machine weights (not needed on the laptop)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--threads", type=int, default=0, help="0 = TensorFlow default; 1 = single thread")
    ap.add_argument("--n-latency", type=int, default=300, help="batch-1 calls timed for latency")
    ap.add_argument("--n-flows", type=int, default=10, help="flows explained for the SHAP-cost measurement")
    ap.add_argument("--nsamples", type=int, default=500)
    a = ap.parse_args()

    import numpy as np
    import tensorflow as tf
    if a.threads:
        tf.config.threading.set_intra_op_parallelism_threads(a.threads)
        tf.config.threading.set_inter_op_parallelism_threads(a.threads)
    import shap
    from shap_utils import build_flow_transformer, build_feature_columns_map, WindowedInstanceExplainer
    from framework.enumerations import EvaluationDatasetSampling
    from revision_eval_fo import DATASETS, dataset_spec, ReorderedModel

    out = dict(label=a.label, dataset=a.dataset, weights=a.weights, threads=a.threads or "default",
               host=platform.node(), cpu=platform.processor() or platform.machine(), os=platform.platform(),
               python=sys.version.split()[0], tensorflow=tf.__version__, shap=shap.__version__,
               nsamples=a.nsamples)
    try:
        with open("/proc/cpuinfo") as fh:
            out["cpu"] = next(l.split(":", 1)[1].strip() for l in fh if l.startswith("model name"))
    except Exception:
        pass
    out["rss_mb_start"] = rss_mb()

    ds = DATASETS[a.dataset]
    ft = build_flow_transformer()
    ft.load_dataset(ds["name"], ds["path"], dataset_spec(a.dataset),
                    evaluation_dataset_sampling=EvaluationDatasetSampling.FilterColumn)
    out["rss_mb_after_data"] = rss_mb()

    m = ft.build_model()
    m.compile(optimizer="adam", loss="binary_crossentropy")
    m.load_weights(a.weights)
    names = list(ft.model_input_spec.feature_names)
    perm = list(range(len(names)))
    M = m
    if a.feature_order:
        M = ReorderedModel(m, names, json.load(open(a.feature_order))["order"])
        perm = M.perm
    out["rss_mb_after_model"] = rss_mb()
    out["params"] = int(m.count_params())
    out["weights_file_kb"] = round(os.path.getsize(a.weights) / 1024, 1)

    # --- order check (same as revision_eval_fo.py --expect) ---
    ref = json.load(open(a.expect))
    ws = ft.parameters.window_size
    num = set(ft.model_input_spec.numeric_feature_names)
    te = np.argwhere(~ft.training_mask).reshape(-1); te = te[te >= ws - 1]
    win = (te[:, None] - (ws - 1)) + np.arange(ws)[None, :]
    inp = []
    for f in names:
        cols_ = [f] if f in num else [c for c in ft.X.columns if str(c).startswith(f)]
        inp.append(ft.X[cols_].values.astype("float32")[win])
    yt = np.asarray(ft.y).astype(str)[te] != ds["benign"]
    pr = M.predict(inp, verbose=0, batch_size=4096).reshape(-1) > 0.5
    cm = (int((pr & yt).sum()), int((pr & ~yt).sum()), int((~pr & ~yt).sum()), int((~pr & yt).sum()))
    exp = (ref["TP"], ref["FP"], ref["TN"], ref["FN"])
    print("VERIFY test-set TP/FP/TN/FN", cm, "stored", exp, "MATCH" if cm == exp else "MISMATCH", flush=True)
    if cm != exp:
        sys.exit("Column order does NOT reproduce the stored test results -- aborting. "
                 "Use the weights file that belongs to this baseline results JSON.")
    out["verify"] = "MATCH"

    # --- batch-1 latency: direct Keras call on one real window, in the model's input order ---
    one = [tf.constant(inp[k][:1]) for k in perm]
    call = tf.function(lambda xs: m(xs, training=False))
    for _ in range(30):
        call(one)
    t = []
    for i in range(a.n_latency):
        j = (i * 37) % len(te)
        xs = [tf.constant(inp[k][j:j + 1]) for k in perm]
        t0 = time.perf_counter(); call(xs).numpy(); t.append((time.perf_counter() - t0) * 1e3)
    t = np.array(t)
    out["latency_ms_batch1_median"] = float(np.median(t))
    out["latency_ms_batch1_p95"] = float(np.percentile(t, 95))

    # --- batch throughput ---
    nb = min(256 * 40, len(te))
    xs = [inp[k][:nb] for k in perm]
    m.predict(xs[:], verbose=0, batch_size=256)
    t0 = time.perf_counter(); m.predict(xs, verbose=0, batch_size=256); dt = time.perf_counter() - t0
    out["throughput_windows_per_s_batch256"] = float(nb / dt)
    del inp, xs
    out["rss_mb_after_inference"] = rss_mb()

    # --- explanation cost: same wrapper, background and flows as the paper ---
    _, idx_map, cols = build_feature_columns_map(ft)
    ex = WindowedInstanceExplainer(ft, M, idx_map, cols)
    Y = np.asarray(ft.y).astype(str)
    test_idx = np.argwhere(~ft.training_mask).reshape(-1)
    safe = test_idx[test_idx - (ws - 1) >= test_idx[0]]
    is_att = Y[safe] != ds["benign"]
    frng = np.random.RandomState(2026)
    chosen = np.concatenate([frng.choice(safe[~is_att], 40, replace=False), frng.choice(safe[is_att], 40, replace=False)])
    chosen = np.concatenate([chosen[: (a.n_flows + 1) // 2], chosen[40: 40 + a.n_flows // 2]])
    train_idx = np.argwhere(ft.training_mask).reshape(-1)
    bg = ft.X.iloc[np.random.RandomState(2026).choice(train_idx, 40, replace=False)][cols].values.astype("float32")
    np.random.seed(2026)
    t_one, t_full = [], []
    num_idx = ex.numeric_column_indices()
    for i1 in chosen:
        f = ex.make_predict_fn(int(i1)); x = ex.flat_vector_for_index(int(i1)).reshape(1, -1)
        e = shap.KernelExplainer(f, bg)
        t0 = time.perf_counter(); e.shap_values(x, nsamples=a.nsamples, silent=True); t1 = time.perf_counter()
        e.shap_values(x, nsamples=a.nsamples, silent=True)  # sigma=0 re-explanation
        for d in range(5):  # 5 perturbed re-explanations (sigma=0.03)
            xn = x.copy(); xn[0, num_idx] += np.random.RandomState(d).normal(0, 0.03, size=len(num_idx)).astype("float32")
            e.shap_values(xn, nsamples=a.nsamples, silent=True)
        t2 = time.perf_counter()
        t_one.append(t1 - t0); t_full.append(t2 - t0)
        print(f"  flow {int(i1)}: one explanation {t1 - t0:.1f}s, full per-flow evaluation {t2 - t0:.1f}s", flush=True)
    out["shap_s_per_explanation_median"] = float(np.median(t_one))
    out["shap_s_per_explanation_max"] = float(np.max(t_one))
    out["full_eval_s_per_flow_median"] = float(np.median(t_full))
    out["n_flows_timed"] = int(len(chosen))
    out["rss_mb_end"] = rss_mb()
    out["rss_mb_peak"] = peak_mb()

    fn = f"edge_cost_{a.dataset}_{a.label}{'_t' + str(a.threads) if a.threads else ''}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print("\n--- SUMMARY", a.dataset, a.label, "threads:", out["threads"], "---")
    for k in ("cpu", "params", "weights_file_kb", "latency_ms_batch1_median", "latency_ms_batch1_p95",
              "throughput_windows_per_s_batch256", "shap_s_per_explanation_median", "full_eval_s_per_flow_median",
              "rss_mb_after_data", "rss_mb_after_model", "rss_mb_peak"):
        v = out[k]
        print(f"{k:36s} {v:.2f}" if isinstance(v, float) else f"{k:36s} {v}")
    print("wrote", fn)


if __name__ == "__main__":
    main()
