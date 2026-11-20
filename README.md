# shap-ft-nids

Faithfulness and stability of SHAP explanations for a lightweight transformer-based network intrusion detector, measured across models retrained with different seeds.

## Overview

The detector is [FlowTransformer](https://github.com/liamdm/FlowTransformer) (Manocchio et al.) in its lightweight configuration: 2 encoder layers and a Last Token head, with 239,649 parameters on NSL-KDD and 679,749 on NF-UNSW-NB15-v2. The model is unchanged apart from two compatibility patches for current pandas and Keras.

It was trained 9 times on NSL-KDD and 7 times on NF-UNSW-NB15-v2, each time with a different seed. Each trained model (an instance) was explained with KernelSHAP on the same 80 test flows and the same 40-flow background, and four metrics were measured:

- **comprehensiveness**: drop in the model's confidence when the top 5 SHAP features are masked
- **sufficiency gap**: change in confidence when only those 5 are kept
- **stability**: cosine similarity between the explanation of a flow and the explanation after adding a bit of Gaussian noise (σ = 0.03)
- **noise floor**: cosine similarity between two explanations of the same unchanged flow, i.e. KernelSHAP's own sampling noise

Across instances, the coefficient of variation of F1 is 2.7% on NSL-KDD and under 0.1% on NF-UNSW-NB15-v2, while that of comprehensiveness is 13.8% and 10.4%. Stability is at the noise floor on both datasets.

## Files

```
.
├── framework/, implementations/   FlowTransformer code (AGPL-3.0, by liamdm), with two compatibility patches
├── prepare_nsl_kdd.py             builds data/NSL-KDD.csv from KDDTrain+.txt and KDDTest+.txt
├── prepare_nf_unsw.py             builds data/NF-UNSW-NB15-v2-prepared.csv from the parquet file
├── baseline_train.py              trains one NSL-KDD model:          python baseline_train.py SEED
├── baseline_train_nf_unsw.py      trains one NF-UNSW-NB15-v2 model:   python baseline_train_nf_unsw.py SEED
├── smoke_test.py                  quick 10-step check that everything runs
├── shap_utils.py                  wrapper so SHAP only changes the flow being explained, not the 7 flows before it
├── test_shap_wrapper*.py          checks the wrapper gives the same prediction as FlowTransformer
├── revision_eval_fo.py            main script: explains the flows and computes all the metrics
├── edge_cost_benchmark.py         measures prediction and explanation time on a CPU
├── make_tables_ieee.py            builds the summary tables from results/ (no TensorFlow needed)
├── feature_orders/                column order each model was trained with (read the section below)
├── results/
│   ├── detection/                 training results for each model (F1, TP/FP/TN/FN, ...)
│   ├── shap_main/                 main runs: 80 flows, 500 SHAP samples
│   ├── sigma_sweep/               20 flows re-explained at σ = 0.01, 0.03, 0.05, 0.1
│   ├── bg_sensitivity/            same 20 flows with a class-balanced background
│   ├── sample_budget/             150 and 1000 SHAP samples on one model
│   ├── edge_cost/                 CPU timing measurements
│   └── derived/                   all summary statistics + the generated tables
├── logs/                          run logs, including every VERIFY ... MATCH line
└── scripts/                       shell scripts as originally run (they assume a flat folder with weights
                                   and result files next to the code)
```

## The models

The prefix shows where a model was trained: `dev` on the original development setup, `sbx` on a cloud VM, `lap` and `sig` on a laptop (Ryzen 5 5500U, CPU only). The number is the seed.

| Dataset | Model | Weights | Training results | Feature order |
|---|---|---|---|---|
| NSL-KDD | dev44 | `baseline_model_seed44.weights.h5` | `baseline_results_seed44.json` | `nsl_dev44.json` |
| NSL-KDD | sbx45 | `baseline_model_seed45.weights.h5` | `baseline_results_seed45.json` | `nsl_laptop.json` |
| NSL-KDD | sbx46 | `baseline_model_seed46.weights.h5` | `baseline_results_seed46.json` | `nsl_laptop.json` |
| NSL-KDD | lap42 | `baseline_model_seed42.weights.h5` | `baseline_results_seed42.json` | `nsl_laptop.json` |
| NSL-KDD | lap47 | `baseline_model_seed47.weights.h5` | `baseline_results_seed47.json` | `nsl_laptop.json` |
| NSL-KDD | lap48 | `baseline_model_seed48.weights.h5` | `baseline_results_seed48.json` | `nsl_laptop.json` |
| NSL-KDD | sig50 | `baseline_model_seed50.weights.h5` | `baseline_results_seed50.json` | `nsl_laptop.json` |
| NSL-KDD | sig51 | `baseline_model_seed51.weights.h5` | `baseline_results_seed51.json` | `nsl_laptop.json` |
| NSL-KDD | sig52 | `baseline_model_seed52.weights.h5` | `baseline_results_seed52.json` | `nsl_laptop.json` |
| NF-UNSW-NB15-v2 | dev43 | `nf_unsw_baseline_model_seed43.weights.h5` | `nf_unsw_baseline_results_seed43.json` | `nfunsw_dev43.json` |
| NF-UNSW-NB15-v2 | dev44 | `nf_unsw_baseline_model_seed44.weights.h5` | `nf_unsw_baseline_results_seed44.json` | `nfunsw_dev44.json` |
| NF-UNSW-NB15-v2 | sbx45 | `nf_unsw_baseline_model_seed45.weights.h5` | `nf_unsw_baseline_results_seed45.json` | `nfunsw_laptop.json` |
| NF-UNSW-NB15-v2 | lap42 | `nf_unsw_baseline_model_seed42.weights.h5` | `nf_unsw_baseline_results_seed42.json` | `nfunsw_laptop.json` |
| NF-UNSW-NB15-v2 | lap46 | `nf_unsw_baseline_model_seed46.weights.h5` | `nf_unsw_baseline_results_seed46.json` | `nfunsw_laptop.json` |
| NF-UNSW-NB15-v2 | sig50 | `nf_unsw_baseline_model_seed50.weights.h5` | `nf_unsw_baseline_results_seed50.json` | `nfunsw_laptop.json` |
| NF-UNSW-NB15-v2 | sig51 | `nf_unsw_baseline_model_seed51.weights.h5` | `nf_unsw_baseline_results_seed51.json` | `nfunsw_laptop.json` |

Each result file has one JSON line per explained flow, with the prediction, the top features, comprehensiveness, sufficiency gap, noise floor (`null_cos`), stability and the full SHAP values. The `.meta.json` next to it lists which flows and background rows were used, the settings, and the library versions.

## Setup

Tested with Python 3.12 on Ubuntu, CPU only.

```bash
git clone https://github.com/priy4n5hu/shap-ft-nids.git
cd shap-ft-nids
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The trained weights are in `weights.zip` on the [Releases](../../releases) page. Unzip it in the repository folder so the `.weights.h5` files are in `weights/`.

The datasets are not included. Create a `data/` folder first.

- **NSL-KDD**: download `KDDTrain+.txt` and `KDDTest+.txt` from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html), put them in `data/` and run `python prepare_nsl_kdd.py`. The official train/test split is used.
- **NF-UNSW-NB15-v2** (from [Sarhan et al., University of Queensland](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)): a cleaned parquet export with 1,986,745 flows was used. Run `python prepare_nf_unsw.py path/to/NF-UNSW-NB15-V2.parquet`. The script keeps the 37 features FlowTransformer uses for this format, samples 30,000 benign flows and up to 4,000 per attack type, does a stratified 80/20 split and shuffles the rows (seed 42).
<!-- TODO before publishing: add the exact download link for the cleaned parquet export. -->

## Input column order

FlowTransformer builds its list of numeric columns from a Python `set` (line 195 of `framework/flow_transformer.py`). Set order depends on Python's hash seed, which changes between runs. The first run saves a cache in `cache/` and later runs reuse its order, but a fresh clone builds a new random order. Keras loads weights without checking column names, so a model fed columns in the wrong order runs without errors but gives wrong predictions.

Every command below therefore passes `--feature-order` with the order the model was trained with, and `--expect`, which recomputes the full test-set confusion matrix and stops if it does not match the one recorded during training. All 16 models print `VERIFY ... MATCH` from a fresh clone with no cache.

The laptop and VM models were trained with a shared cache and use the same order (`nsl_laptop.json`, `nfunsw_laptop.json`). The orders of the three `dev` models were recovered from their saved outputs.

## Usage

Summary tables from the stored results (a few seconds, no TensorFlow needed):

```bash
python make_tables_ieee.py
```

This prints the per-model and overall numbers, writes them to `results/derived/tables_ieee_corrected.tex` and checks them against `results/derived/stats_v2.json`.

Check that a model loads correctly (under a minute):

```bash
python revision_eval_fo.py --dataset nsl --label lap47 \
    --weights weights/baseline_model_seed47.weights.h5 \
    --expect results/detection/baseline_results_seed47.json \
    --feature-order feature_orders/nsl_laptop.json --verify-only
```

Re-run the full explanation for that model by dropping `--verify-only` and setting a new output folder. The script resumes from existing files in `--out-dir`, so do not point it at the stored results. On a laptop CPU this takes about 30 minutes for an NSL-KDD model and about 70 for an NF-UNSW-NB15-v2 model.

```bash
python revision_eval_fo.py --dataset nsl --label lap47 \
    --weights weights/baseline_model_seed47.weights.h5 \
    --expect results/detection/baseline_results_seed47.json \
    --feature-order feature_orders/nsl_laptop.json --out-dir my_results
```

For NF-UNSW-NB15-v2, use `--dataset nfunsw` and the `nf_unsw_*` files. The defaults are the settings of the main runs (`--n-per-class 40 --nsamples 500 --topk 5 --sigmas 0.03 --bg-mode random`). The additional checks use:

| Check | Options |
|---|---|
| σ sweep | `--n-per-class 10 --sigmas 0.01,0.03,0.05,0.1` |
| class-balanced background | `--n-per-class 10 --bg-mode stratified` |
| number of SHAP samples | `--n-per-class 10 --nsamples 150` or `--nsamples 1000` |

KernelSHAP is stochastic, so re-runs give values close to the stored ones but not identical.

Time prediction and explanation on a CPU:

```bash
python edge_cost_benchmark.py --dataset nsl --label lap47 \
    --weights weights/baseline_model_seed47.weights.h5 \
    --expect results/detection/baseline_results_seed47.json \
    --feature-order feature_orders/nsl_laptop.json        # add --threads 1 for single-thread
```

Train a new model (2 to 8 minutes on a laptop CPU):

```bash
python baseline_train.py 60          # NSL-KDD, seed 60
python baseline_train_nf_unsw.py 60  # NF-UNSW-NB15-v2, seed 60
```

Multi-threaded CPU training is not bit-reproducible, so the same seed does not give identical weights. Models trained and evaluated in the same folder share `cache/` and do not need `--feature-order`.

## License

AGPL-3.0, inherited from FlowTransformer (see `LICENSE`). The code in `framework/` and `implementations/` is by the FlowTransformer authors.

## Citation

If you use this code, please cite FlowTransformer:

> L. D. Manocchio, S. Layeghy, W. W. Lo, G. K. Kulatilleke, M. Sarhan, and M. Portmann, "FlowTransformer: A transformer framework for flow-based network intrusion detection systems," *Expert Systems with Applications*, vol. 241, Art. no. 122564, 2024.
