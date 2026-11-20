#!/bin/bash
# Background-sensitivity check (reviewer comment 1.5) -- laptop-only, 9 models.
#
# Reviewer 1.5 asks us to assess sensitivity to how the KernelSHAP background was
# selected. The background is currently a plain random 40-row draw (not stratified
# by class) -- already stated explicitly in the response letter. This script re-runs
# the SAME 9 models used for the sigma sweep, SAME n=20/model, SAME flow-seed=2026
# (identical explained flows), but with --bg-mode stratified (20 benign + 20 attack
# background rows instead of a random 40) and a SINGLE sigma (0.03, the paper's
# actual value -- no need for the full 4-sigma sweep here, this check is about the
# background, not perturbation scale).
#
# This is directly comparable, flow-for-flow, to the sigma-sweep's sigma=0.03 column
# you already have (same 9 models, same n=20, same flows -- only the background mode
# differs). Output goes to NEW files -- the tag includes "bgstratified" instead of
# "bgrandom", so this will NOT touch or overwrite any existing results.
#
# Cost: single sigma is much cheaper than the 4-sigma sweep you just ran (roughly
# 1/3 the KernelSHAP calls per flow). Expect roughly 1.5-2 hours total on this
# laptop, going by last night's timing. Safe to run overnight or just leave running.
#
# If any weights file below gives "file not found", tell me the actual filename --
# don't guess or skip a line.

run() { echo "=== START $* $(date) ==="; python revision_eval.py "$@" --n-per-class 10 --sigmas 0.03 --bg-mode stratified; echo "=== END $* rc=$? $(date) ==="; }

# --- 5 freshly-trained instances (already trained -- same weights as the sigma sweep) ---
run --dataset nsl    --weights baseline_model_seed50.weights.h5        --label sig50
run --dataset nsl    --weights baseline_model_seed51.weights.h5        --label sig51
run --dataset nsl    --weights baseline_model_seed52.weights.h5        --label sig52
run --dataset nfunsw --weights nf_unsw_baseline_model_seed50.weights.h5 --label sig50
run --dataset nfunsw --weights nf_unsw_baseline_model_seed51.weights.h5 --label sig51

# --- your 4 laptop-trained models from the n=80 run ---
run --dataset nsl    --weights baseline_model_seed47.weights.h5        --label lap47
run --dataset nsl    --weights baseline_model_seed48.weights.h5        --label lap48
run --dataset nfunsw --weights nf_unsw_baseline_model_seed42.weights.h5 --label lap42
run --dataset nfunsw --weights nf_unsw_baseline_model_seed46.weights.h5 --label lap46

echo "=== ALL DONE $(date) ==="
