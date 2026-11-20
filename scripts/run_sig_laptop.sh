#!/bin/bash
# Run from your project folder with the venv active:   bash run_sig_laptop.sh
# Step 1 checks that 3 laptop models reproduce their stored test results.
# Step 2 runs SHAP + LIME (n=80) for the 5 extra laptop instances (seeds 50-52 NSL-KDD, 50-51 NF-UNSW-NB15-v2).
# Step 3 zips everything into sig_results.zip  -> send that file back.
LOG=run_sig_laptop.log
echo "=== run started $(date) ===" | tee -a $LOG
need() { for f in "$@"; do [ -f "$f" ] || echo "MISSING FILE: $f" | tee -a $LOG; done; }
need baseline_model_seed47.weights.h5 baseline_results_seed47.json baseline_model_seed48.weights.h5 baseline_results_seed48.json \
     nf_unsw_baseline_model_seed46.weights.h5 nf_unsw_baseline_results_seed46.json \
     baseline_model_seed50.weights.h5 baseline_results_seed50.json baseline_model_seed51.weights.h5 baseline_results_seed51.json \
     baseline_model_seed52.weights.h5 baseline_results_seed52.json \
     nf_unsw_baseline_model_seed50.weights.h5 nf_unsw_baseline_results_seed50.json \
     nf_unsw_baseline_model_seed51.weights.h5 nf_unsw_baseline_results_seed51.json

echo "--- Step 1: verify ---" | tee -a $LOG
for s in 47 48; do
  python revision_eval_fo.py --dataset nsl --weights baseline_model_seed$s.weights.h5 --label lap$s --expect baseline_results_seed$s.json --verify-only 2>&1 | grep -E "VERIFY|abort|Error" | sed "s/^/nsl seed $s: /" | tee -a $LOG
done
python revision_eval_fo.py --dataset nfunsw --weights nf_unsw_baseline_model_seed46.weights.h5 --label lap46 --expect nf_unsw_baseline_results_seed46.json --verify-only 2>&1 | grep -E "VERIFY|abort|Error" | sed "s/^/nfunsw seed 46: /" | tee -a $LOG

echo "--- Step 2: SHAP + LIME ---" | tee -a $LOG
run() { echo "=== START $* $(date) ===" | tee -a $LOG; python "$@" 2>&1 | grep -E "VERIFY|abort|Error|\(80/80\)" | tee -a $LOG; echo "=== END $(date) ===" | tee -a $LOG; }
for s in 50 51 52; do
  run revision_eval_fo.py --dataset nsl --weights baseline_model_seed$s.weights.h5 --label sig$s --expect baseline_results_seed$s.json --out-dir results_rev_fixed
  run lime_revision_eval_fo.py --dataset nsl --weights baseline_model_seed$s.weights.h5 --label sig$s --expect baseline_results_seed$s.json --out-dir results_rev_lime_fixed
done
for s in 50 51; do
  run revision_eval_fo.py --dataset nfunsw --weights nf_unsw_baseline_model_seed$s.weights.h5 --label sig$s --expect nf_unsw_baseline_results_seed$s.json --out-dir results_rev_fixed
  run lime_revision_eval_fo.py --dataset nfunsw --weights nf_unsw_baseline_model_seed$s.weights.h5 --label sig$s --expect nf_unsw_baseline_results_seed$s.json --out-dir results_rev_lime_fixed
done

echo "--- Step 3: zip ---" | tee -a $LOG
python - <<'PY'
import zipfile, glob, os
files = glob.glob("results_rev_fixed/*") + glob.glob("results_rev_lime_fixed/*") + ["run_sig_laptop.log"]
files += [f for f in ["baseline_results_seed47.json", "baseline_results_seed48.json", "baseline_results_seed50.json",
                      "baseline_results_seed51.json", "baseline_results_seed52.json", "nf_unsw_baseline_results_seed46.json",
                      "nf_unsw_baseline_results_seed50.json", "nf_unsw_baseline_results_seed51.json"] if os.path.exists(f)]
with zipfile.ZipFile("sig_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f)
print("wrote sig_results.zip with", len(files), "files")
PY
echo "=== run finished $(date) ===" | tee -a $LOG
