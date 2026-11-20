"""Regenerate the corrected numbers for the IEEEtran paper's Tables II-IV and Figs. 1-2
from the raw result files in this repository (no TensorFlow needed).

    python3 make_tables_ieee.py            # writes results/derived/tables_ieee_corrected.tex and prints a summary

Protocol behind every number (the revised, order-verified protocol):
  * 80 test flows per dataset (40 benign + 40 attack, flow-seed 2026), identical for every instance
  * one 40-flow random training background (bg-seed 2026) for KernelSHAP AND for masking
  * KernelSHAP nsamples=500 (SHAP default l1_reg="num_features(10)"), top-k=5
  * stability: cosine similarity, sigma=0.03 on numeric features, mean of 5 draws;
    noise floor: one re-explanation of the unperturbed input (sigma=0)
  * every instance reproduces its stored test-set confusion matrix (see logs/verify_runs.log,
    logs/run_sig_laptop.log)
Conventions (same as the revised ICMLDE manuscript): per-flow +/- is the population SD over the
80 flows; cross-instance +/- is the sample SD (ddof=1); saturation fraction is rounded half-up.
"""
import json, os, glob
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

K = os.path.dirname(os.path.abspath(__file__))
NSL = ["dev44", "sbx45", "sbx46", "lap42", "lap47", "lap48", "sig50", "sig51", "sig52"]
NF = ["dev43", "dev44", "sbx45", "lap42", "lap46", "sig50", "sig51"]
MACH = {"dev": "dev.", "sbx": "sbx.", "lap": "laptop", "sig": "laptop"}
DET = {"nsl": "baseline_results_seed", "nfunsw": "nf_unsw_baseline_results_seed"}


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def hu(x, nd=2):
    return str(Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def sd1(x):
    return float(np.std(x, ddof=1))


def inst(ds, lab):
    rr = rows(os.path.join(K, f"results/shap_main/{ds}_{lab}_ns500_bgrandom2026.jsonl"))
    a = lambda k: np.array([r[k] for r in rr], float)
    comp, suff = a("comprehensiveness"), np.abs(a("sufficiency_gap"))
    stab = np.array([np.nanmean(r["stability"]["0.03"]) for r in rr])
    null = np.array([np.nanmean(r["null_cos"]) for r in rr])
    det = json.load(open(os.path.join(K, f"results/detection/{DET[ds]}{lab[3:]}.json")))
    return dict(n=len(rr), f1=det["f1"], rec=det.get("detection_rate_recall"), far=det.get("false_alarm_rate"),
                comp=comp.mean(), comp_sd=comp.std(), suff=suff.mean(), suff_sd=suff.std(),
                stab=np.nanmean(stab), stab_sd=np.nanstd(stab), null=np.nanmean(null),
                sat=np.mean([r["saturated"] for r in rr]), rows=[r["row_index"] for r in rr])


D = {("nsl", l): inst("nsl", l) for l in NSL}
D.update({("nfunsw", l): inst("nfunsw", l) for l in NF})
ref = D[("nsl", NSL[0])]["rows"]
assert all(D[k]["rows"] == ref for k in D if k[0] == "nsl"), "NSL instances must share the same 80 flows"
assert all(D[k]["rows"] == D[("nfunsw", NF[0])]["rows"] for k in D if k[0] == "nfunsw")


def agg(ds, labs, key):
    v = np.array([D[(ds, l)][key] for l in labs], float)
    return v.mean(), sd1(v), v


out = []
# ---------------- Table II: detection (NSL-KDD, 9 instances) ----------------
f1m, f1s, _ = agg("nsl", NSL, "f1"); rm, rs, _ = agg("nsl", NSL, "rec"); am, as_, _ = agg("nsl", NSL, "far")
out += [f"% ===== Table II (detection, NSL-KDD KDDTest+, mean +/- std over {len(NSL)} runs) =====",
        "Parameters & 239,649 \\\\", f"F1 & ${f1m:.3f} \\pm {f1s:.3f}$ \\\\",
        f"Detection rate (recall) & ${rm:.3f} \\pm {rs:.3f}$ \\\\", f"False alarm rate & ${am:.3f} \\pm {as_:.3f}$ \\\\", ""]
# ---------------- Table III: per instance, NSL-KDD ----------------
out.append("% ===== Table III (NSL-KDD, SHAP, k=5, n=80 flows/instance) =====")
out.append("% Model & Det. F1 & Comprehensiveness & Suff. gap & Stability & Noise floor & Sat. frac. \\\\")
for l in NSL:
    m = D[("nsl", l)]
    out.append(f"Seed {l[3:]} ({MACH[l[:3]]}) & {m['f1']:.3f} & ${m['comp']:+.3f} \\pm {m['comp_sd']:.3f}$ & "
               f"${m['suff']:.3f} \\pm {m['suff_sd']:.3f}$ & ${m['stab']:.3f} \\pm {m['stab_sd']:.3f}$ & ${m['null']:.3f}$ & {hu(m['sat'])} \\\\")
cm, cs, C = agg("nsl", NSL, "comp"); sm, ss, St = agg("nsl", NSL, "stab"); nm, ns, _ = agg("nsl", NSL, "null")
out += ["\\midrule", f"\\emph{{Cross-model (n={len(NSL)})}} & ${f1m:.3f} \\pm {f1s:.3f}$ & ${cm:+.3f} \\pm {cs:.3f}$ & -- & "
        f"${sm:.3f} \\pm {ss:.3f}$ & ${nm:.3f} \\pm {ns:.3f}$ & -- \\\\", ""]
# ---------------- Table IV: cross-model, both datasets ----------------
uf, ufs, UF = agg("nfunsw", NF, "f1"); uc, ucs, UC = agg("nfunsw", NF, "comp"); us, uss, US = agg("nfunsw", NF, "stab"); un, uns, _ = agg("nfunsw", NF, "null")
out += [f"% ===== Table IV (cross-model: NF-UNSW-NB15-v2 n={len(NF)} vs NSL-KDD n={len(NSL)}) =====",
        "% Dataset & Det. F1 & Comprehensiveness & Stability & Noise floor \\\\",
        f"NF-UNSW-NB15-v2 & ${uf:.4f}\\pm{ufs:.4f}$ & ${uc:+.3f}\\pm{ucs:.3f}$ & ${us:.3f}\\pm{uss:.3f}$ & ${un:.3f}\\pm{uns:.3f}$ \\\\",
        f"NSL-KDD & ${f1m:.3f}\\pm{f1s:.3f}$ & ${cm:+.3f}\\pm{cs:.3f}$ & ${sm:.3f}\\pm{ss:.3f}$ & ${nm:.3f}\\pm{ns:.3f}$ \\\\", ""]
# ---------------- Figures ----------------
tick = {"dev": "D", "sbx": "S", "lap": "L", "sig": "L"}
out += ["% ===== Fig. 1 (NSL-KDD per instance) =====",
        "% xticklabels={" + ",".join(tick[l[:3]] + l[3:] for l in NSL) + "}",
        "% F1:   " + " ".join(f"({i + 1},{D[('nsl', l)]['f1']:.3f})" for i, l in enumerate(NSL)),
        "% comp: " + " ".join(f"({i + 1},{D[('nsl', l)]['comp']:.3f})" for i, l in enumerate(NSL)),
        "% stab: " + " ".join(f"({i + 1},{D[('nsl', l)]['stab']:.3f})" for i, l in enumerate(NSL)), "",
        "% ===== Fig. 2 (saturation fraction vs stability / comprehensiveness) ====="]
for ds, labs in (("nsl", NSL), ("nfunsw", NF)):
    out.append(f"% {ds} stab: " + " ".join(f"({hu(D[(ds, l)]['sat'])},{D[(ds, l)]['stab']:.3f})" for l in labs))
    out.append(f"% {ds} comp: " + " ".join(f"({hu(D[(ds, l)]['sat'])},{D[(ds, l)]['comp']:.3f})" for l in labs))
cv = lambda v: 100 * sd1(v) / np.mean(v)
out += ["", f"% CV F1: NSL {cv(agg('nsl', NSL, 'f1')[2]):.1f}% | NF {cv(UF):.2f}%   CV comprehensiveness: NSL {cv(C):.1f}% | NF {cv(UC):.1f}%   "
            f"CV stability: NSL {cv(St):.1f}% | NF {cv(US):.1f}%"]
open(os.path.join(K, "results/derived/tables_ieee_corrected.tex"), "w").write("\n".join(out) + "\n")
print("\n".join(out))

# ---------------- self-check against the stats file used for the ICMLDE manuscript ----------------
ST = json.load(open(os.path.join(K, "results/derived/stats_v2.json")))
for ds, labs in (("nsl", NSL), ("nfunsw", NF)):
    X = ST[ds]
    for key, mine in (("f1_mean", agg(ds, labs, "f1")[0]), ("comp_mean", agg(ds, labs, "comp")[0]), ("comp_sd", agg(ds, labs, "comp")[1]),
                      ("stab_mean", agg(ds, labs, "stab")[0]), ("null_mean", agg(ds, labs, "null")[0])):
        assert abs(X[key] - mine) < 1e-12, (ds, key, X[key], mine)
    for l in labs:
        for key in ("comp", "suff", "stab", "null", "sat"):
            assert abs(ST["inst"][f"{ds}:{l}"][key] - D[(ds, l)][key]) < 1e-12, (ds, l, key)
print("\nSelf-check passed: every value equals results/derived/stats_v2.json (the ICMLDE manuscript's source).")
