"""
ICMLDE revision (paper 1906): unified SHAP explanation + faithfulness +
stability evaluation for ONE trained model instance.

Replaces the pair shap_explain_day3.py -> day4_faithfulness_stability.py
(and the NF-UNSW mirrors). Same windowed-instance wrapper (shap_utils.py),
same metric definitions (top-k masking to background mean; cosine
similarity of 41-dim aggregated SHAP vectors). Four fixes relative to the
original scripts, all found while preparing the revision:

 1. ONE background sample, drawn with --bg-seed and independent of the
    model, is used for KernelSHAP, for masking, AND for the stability
    re-explanations. (The original Day 3 and Day 4 scripts drew two
    disjoint 40-row backgrounds -- 0/40 overlap -- contradicting the
    paper's statement that masking uses "the same 40-flow background
    sample used for KernelSHAP itself".)
 2. The SAME explained flows for every model (chosen with --flow-seed,
    independent of the model's training seed), so the cross-model
    comparison is not confounded by different flow samples. (Originally
    seed 42/43/44 models were explained on three disjoint flow sets.)
 3. KernelSHAP budget is configurable (--nsamples, default 500) and a
    sigma=0 control is recorded for every flow: the identical input is
    re-explained and compared to the original. At nsamples=150 this
    control alone gave mean cosine 0.79 on the seed-42 model -- i.e. most
    of the originally reported "instability" was KernelSHAP's own
    sampling noise. At 500 it is ~0.97, at 1000 ~0.98.
 4. Perturbation noise for flow i is seeded from (flow-seed, row, sigma,
    draw), so every model sees identical noise vectors for the same flow.

Output is resumable: one JSON line per flow in <out-dir>/<tag>.jsonl.
Re-running the same command skips flows already written, so an
interrupted overnight run just continues.

Usage (from the project folder, venv active):
  python revision_eval.py --dataset nsl    --weights baseline_model_seed47.weights.h5        --label lap47
  python revision_eval.py --dataset nfunsw --weights nf_unsw_baseline_model_seed46.weights.h5 --label lap46
Quick smoke test (~1 min):
  python revision_eval.py --dataset nsl --weights baseline_model_seed47.weights.h5 --label smoke --limit 2 --nsamples 150 --n-pert 1
"""
import argparse
import json
import os
import platform
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import shap

import shap_utils
from shap_utils import build_feature_columns_map, WindowedInstanceExplainer, build_flow_transformer
from framework.dataset_specification import NamedDatasetSpecifications
from framework.enumerations import EvaluationDatasetSampling

DATASETS = {
    "nsl": dict(name="NSL-KDD", path=os.path.join("data", "NSL-KDD.csv"), benign="normal"),
    "nfunsw": dict(name="NF-UNSW-NB15-v2", path=os.path.join("data", "NF-UNSW-NB15-v2-prepared.csv"), benign="Benign"),
}


def dataset_spec(key):
    if key == "nsl":
        return NamedDatasetSpecifications.nsl_kdd
    from prepare_nf_unsw import nf_unsw_nb15_v2
    return nf_unsw_nb15_v2


class BatchedExplainer(WindowedInstanceExplainer):
    """Same windowing as shap_utils.WindowedInstanceExplainer; only the
    prediction batch size is larger (pure speed-up, identical outputs)."""

    def __init__(self, *a, batch_size=4096, **kw):
        super().__init__(*a, **kw)
        self.batch_size = batch_size

    def make_predict_fn(self, i1):
        context = self.context_for_index(i1)
        model, names, idx_map, bs = self.model, self.feature_names, self.feature_col_idx_map, self.batch_size

        def predict_fn(Z):
            n = Z.shape[0]
            inputs = []
            for f in names:
                tv = Z[:, idx_map[f]].astype("float32").reshape(n, 1, -1)
                ctx = np.broadcast_to(context[f], (n,) + context[f].shape)
                inputs.append(np.concatenate([ctx, tv], axis=1))
            return model.predict(inputs, verbose=0, batch_size=bs).reshape(-1)

        return predict_fn


def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")



class ReorderedModel:
    """Feeds the numeric inputs in the order the weights were TRAINED with.
    FlowTransformer builds its numeric feature list from a Python set, so the
    order depends on the process's hash seed unless a preprocessing cache is
    shared; weights trained in another environment must be fed in their own
    training order. Categorical inputs (fixed spec order) are unchanged."""

    def __init__(self, m, names, num_order):
        self.m = m
        n_num = len(num_order)
        assert sorted(num_order) == sorted(names[:n_num]), "order must permute the numeric features"
        self.perm = [names.index(f) for f in num_order] + list(range(n_num, len(names)))

    def predict(self, inputs, **kw):
        return self.m.predict([inputs[k] for k in self.perm], **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--verify-only", action="store_true", help="only run the --expect check, then exit")
    ap.add_argument("--expect", default=None, help="baseline results JSON; abort unless the model reproduces its stored test TP/FP/TN/FN")
    ap.add_argument("--feature-order", default=None, help="JSON with {\"order\": [numeric feature names in training order]}")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--label", required=True, help="model instance name, e.g. dev42, lap47, sbx45")
    ap.add_argument("--n-per-class", type=int, default=40, help="flows per class (40 -> n=80 total)")
    ap.add_argument("--flow-seed", type=int, default=2026)
    ap.add_argument("--bg-seed", type=int, default=2026)
    ap.add_argument("--bg-size", type=int, default=40)
    ap.add_argument("--bg-mode", choices=["random", "stratified"], default="random")
    ap.add_argument("--nsamples", type=int, default=500)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--sigmas", default="0.03", help="comma list, e.g. 0.01,0.03,0.05,0.1")
    ap.add_argument("--n-pert", type=int, default=5)
    ap.add_argument("--n-null", type=int, default=1, help="sigma=0 re-explanations per flow (estimator noise floor)")
    ap.add_argument("--limit", type=int, default=0, help="only first N flows (smoke test)")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out-dir", default="results_rev")
    a = ap.parse_args()

    sigmas = [float(s) for s in a.sigmas.split(",") if s.strip()]
    ds = DATASETS[a.dataset]
    tag = (f"{a.dataset}_{a.label}_ns{a.nsamples}_bg{a.bg_mode}{a.bg_seed}"
           + ("" if sigmas == [0.03] else "_sig" + "-".join(str(s) for s in sigmas)))
    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, tag + ".jsonl")
    meta_path = os.path.join(a.out_dir, tag + ".meta.json")

    t0 = time.time()
    np.random.seed(a.flow_seed)  # KernelSHAP's internal coalition sampling uses the global RNG

    ft = build_flow_transformer()
    ft.load_dataset(ds["name"], ds["path"], dataset_spec(a.dataset),
                    evaluation_dataset_sampling=EvaluationDatasetSampling.FilterColumn)
    m = ft.build_model()
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])
    m.load_weights(a.weights)
    if a.feature_order:
        m = ReorderedModel(m, list(ft.model_input_spec.feature_names), json.load(open(a.feature_order))["order"])
        print("feature order from", a.feature_order, flush=True)
    if a.expect:
        ref = json.load(open(a.expect))
        ws_ = ft.parameters.window_size
        names_ = list(ft.model_input_spec.feature_names)
        num_ = set(ft.model_input_spec.numeric_feature_names)
        te = np.argwhere(~ft.training_mask).reshape(-1); te = te[te >= ws_ - 1]
        win = (te[:, None] - (ws_ - 1)) + np.arange(ws_)[None, :]
        inp = []
        for f_ in names_:
            cols_ = [f_] if f_ in num_ else [c for c in ft.X.columns if str(c).startswith(f_)]
            inp.append(ft.X[cols_].values.astype("float32")[win])
        yt = np.asarray(ft.y).astype(str)[te] != ds["benign"]
        pr = m.predict(inp, verbose=0, batch_size=4096).reshape(-1) > 0.5
        cm = (int((pr & yt).sum()), int((pr & ~yt).sum()), int((~pr & ~yt).sum()), int((~pr & yt).sum()))
        exp = (ref["TP"], ref["FP"], ref["TN"], ref["FN"])
        print("VERIFY test-set TP/FP/TN/FN", cm, "stored", exp, "MATCH" if cm == exp else "MISMATCH", flush=True)
        if cm != exp:
            sys.exit("Feature order does NOT reproduce the stored test results -- aborting.")
        del inp
    if a.verify_only:
        sys.exit(0)
    _, idx_map, cols = build_feature_columns_map(ft)
    expl = BatchedExplainer(ft, m, idx_map, cols, batch_size=a.batch)
    names = list(ft.model_input_spec.feature_names)
    num_idx = expl.numeric_column_indices()
    Y = np.asarray(ft.y).astype(str)
    print(f"[{time.time()-t0:.1f}s] {ds['name']} loaded, weights={a.weights}, "
          f"{len(names)} features / {len(cols)} flat columns ({len(num_idx)} numeric)", flush=True)

    # --- explained flows: identical for every model ---
    ws = ft.parameters.window_size
    test_idx = np.argwhere(~ft.training_mask).reshape(-1)
    safe = test_idx[test_idx - (ws - 1) >= test_idx[0]]
    is_att = Y[safe] != ds["benign"]
    frng = np.random.RandomState(a.flow_seed)
    chosen = np.concatenate([
        frng.choice(safe[~is_att], a.n_per_class, replace=False),
        frng.choice(safe[is_att], a.n_per_class, replace=False),
    ])
    if a.limit:
        chosen = np.concatenate([chosen[: (a.limit + 1) // 2], chosen[a.n_per_class: a.n_per_class + a.limit // 2]])

    # --- ONE background, used everywhere ---
    train_idx = np.argwhere(ft.training_mask).reshape(-1)
    brng = np.random.RandomState(a.bg_seed)
    if a.bg_mode == "random":
        bg_idx = brng.choice(train_idx, a.bg_size, replace=False)
    else:
        tr_att = Y[train_idx] != ds["benign"]
        h = a.bg_size // 2
        bg_idx = np.concatenate([brng.choice(train_idx[~tr_att], a.bg_size - h, replace=False),
                                 brng.choice(train_idx[tr_att], h, replace=False)])
    background = ft.X.iloc[bg_idx][cols].values.astype("float32")
    bg_mean = background.mean(axis=0)

    meta = dict(dataset=ds["name"], label=a.label, weights=a.weights, n_flows=int(len(chosen)),
                flow_seed=a.flow_seed, bg_seed=a.bg_seed, bg_mode=a.bg_mode, bg_size=a.bg_size,
                bg_indices=[int(i) for i in bg_idx], bg_attack_frac=float(np.mean(Y[bg_idx] != ds["benign"])),
                chosen=[int(i) for i in chosen], nsamples=a.nsamples, topk=a.topk, sigmas=sigmas,
                n_pert=a.n_pert, n_null=a.n_null, host=platform.node(),
                versions=dict(python=sys.version.split()[0], shap=shap.__version__, numpy=np.__version__))
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        for k in ("chosen", "bg_indices", "nsamples", "topk", "sigmas", "n_pert"):
            if old.get(k) != meta[k] and not a.limit:
                sys.exit(f"Existing {meta_path} has a different '{k}' -- use another --out-dir or --label.")
    json.dump(meta, open(meta_path, "w"), indent=1)

    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["row_index"] for l in open(out_path) if l.strip()}
    todo = [i for i in chosen if int(i) not in done]
    print(f"[{time.time()-t0:.1f}s] {len(chosen)} flows ({len(done)} already done), background attack "
          f"fraction {meta['bg_attack_frac']:.2f} -> {out_path}", flush=True)

    def agg(sv):
        sv = np.asarray(sv).reshape(-1)
        return np.array([float(sv[idx_map[f]].sum()) for f in names])

    for k, i1 in enumerate(todo):
        tf0 = time.time()
        i1 = int(i1)
        f = expl.make_predict_fn(i1)
        x = expl.flat_vector_for_index(i1)
        p = float(f(x.reshape(1, -1))[0])
        pred_att = p >= 0.5
        cls = lambda q: q if pred_att else 1.0 - q
        true_att = bool(Y[i1] != ds["benign"])

        e = shap.KernelExplainer(f, background)
        phi = agg(e.shap_values(x.reshape(1, -1), nsamples=a.nsamples, silent=True))
        order = np.argsort(-np.abs(phi))
        top = [names[j] for j in order[: a.topk]]
        top_cols = sorted(sum((idx_map[t] for t in top), []))

        xc = x.copy(); xc[top_cols] = bg_mean[top_cols]
        comp = cls(p) - cls(float(f(xc.reshape(1, -1))[0]))
        xs = bg_mean.copy(); xs[top_cols] = x[top_cols]
        suff = cls(p) - cls(float(f(xs.reshape(1, -1))[0]))

        null = [cosine(phi, agg(e.shap_values(x.reshape(1, -1), nsamples=a.nsamples, silent=True)))
                for _ in range(a.n_null)]
        stab = {}
        for s in sigmas:
            sims = []
            for d in range(a.n_pert):
                nrng = np.random.RandomState([a.flow_seed, i1 % (2**31), int(round(s * 1e4)), d])
                xn = x.copy()
                xn[num_idx] += nrng.normal(0, s, size=len(num_idx)).astype("float32")
                sims.append(cosine(phi, agg(e.shap_values(xn.reshape(1, -1), nsamples=a.nsamples, silent=True))))
            stab[str(s)] = sims

        rec = dict(row_index=i1, true_label=str(Y[i1]), true_is_attack=true_att, pred_prob_attack=p,
                   pred_is_attack=bool(pred_att), correct=bool(pred_att == true_att),
                   saturated=bool(p < 0.01 or p > 0.99), base_value=float(np.ravel(e.expected_value)[0]),
                   top_features=top, comprehensiveness=comp, sufficiency_gap=suff,
                   null_cos=null, stability=stab, all_feature_shap=dict(zip(names, phi.tolist())),
                   seconds=round(time.time() - tf0, 1))
        with open(out_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        s03 = np.nanmean(stab.get("0.03", next(iter(stab.values())) if stab else [np.nan]))
        print(f"[{time.time()-t0:.0f}s] ({len(done)+k+1}/{len(chosen)}) row={i1} true={Y[i1]:>14s} "
              f"P(att)={p:.3f} comp={comp:+.3f} suff={abs(suff):.3f} null={np.nanmean(null):.3f} "
              f"stab={s03:.3f} ({rec['seconds']}s)", flush=True)

    rows = [json.loads(l) for l in open(out_path) if l.strip()]
    comp = [r["comprehensiveness"] for r in rows]
    suff = [abs(r["sufficiency_gap"]) for r in rows]
    null = [np.nanmean(r["null_cos"]) for r in rows]
    print(f"\n--- SUMMARY {ds['name']} {a.label} (n={len(rows)}, nsamples={a.nsamples}, k={a.topk}) ---")
    print(f"comprehensiveness  {np.mean(comp):+.4f} +- {np.std(comp):.4f}")
    print(f"sufficiency gap    {np.mean(suff):.4f} +- {np.std(suff):.4f}")
    print(f"null (sigma=0)     {np.nanmean(null):.4f} +- {np.nanstd(null):.4f}   <- KernelSHAP noise floor")
    for s in sigmas:
        v = [np.nanmean(r["stability"][str(s)]) for r in rows]
        print(f"stability s={s:<5} {np.nanmean(v):.4f} +- {np.nanstd(v):.4f}")
    print(f"saturation frac    {np.mean([r['saturated'] for r in rows]):.2f}   accuracy on explained flows "
          f"{np.mean([r['correct'] for r in rows]):.2f}")
    print(f"wrote {out_path}  [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
