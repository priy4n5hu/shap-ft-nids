#!/bin/bash
# Sigma sensitivity sweep (reviewer comment 2.4) -- 9-model version, laptop-only.
# Replaces the 14-model plan: the sandbox proved unreliable (died twice, ~20min
# then ~13min), and the file transfer of its 10 pre-trained weight files failed
# on the user's end (zip download not working). Workaround: train 5 fresh local
# instances instead of transferring sandbox weights, combine with the 4 models
# already trained locally this revision (lap47, lap48, nfunsw-lap42, nfunsw-lap46).
#
# Design: n-per-class=10 (n=20/model, reduced from the primary n=80) x sigmas
# 0.01/0.03/0.05/0.1. Same fixed flow-seed/bg-seed=2026 as everything else this
# revision. Output goes to NEW files (tag gets a _sig... suffix automatically) --
# this will NOT touch or overwrite your existing n=80 results.
#
# Run this AFTER training seeds 50,51,52 (NSL) and 50,51 (NF-UNSW) -- see chat
# instructions. Expected total time: ~2-2.5 hours. Fine to run live or overnight.

run() { echo "=== START $* $(date) ==="; python revision_eval.py "$@" --n-per-class 10 --sigmas 0.01,0.03,0.05,0.1; echo "=== END $* rc=$? $(date) ==="; }

# --- 5 freshly-trained instances (train these first, see chat) ---
run --dataset nsl    --weights baseline_model_seed50.weights.h5        --label sig50
run --dataset nsl    --weights baseline_model_seed51.weights.h5        --label sig51
run --dataset nsl    --weights baseline_model_seed52.weights.h5        --label sig52
run --dataset nfunsw --weights nf_unsw_baseline_model_seed50.weights.h5 --label sig50
run --dataset nfunsw --weights nf_unsw_baseline_model_seed51.weights.h5 --label sig51

# --- your 4 laptop-trained models from the n=80 run (if any "file not found",
#     tell me the actual filename you used before -- don't guess) ---
run --dataset nsl    --weights baseline_model_seed47.weights.h5        --label lap47
run --dataset nsl    --weights baseline_model_seed48.weights.h5        --label lap48
run --dataset nfunsw --weights nf_unsw_baseline_model_seed42.weights.h5 --label lap42
run --dataset nfunsw --weights nf_unsw_baseline_model_seed46.weights.h5 --label lap46

echo "=== ALL DONE $(date) ==="
