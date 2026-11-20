# Faithfulness and Stability of SHAP Explanations in Lightweight Transformer-Based NIDS

Code and results for the paper:

> Priyanshu and Himanshu Verma, "Faithfulness and Stability of SHAP Explanations in Lightweight Transformer-Based Network Intrusion Detection for Edge Deployment," *12th IEEE International Symposium on Smart Electronic Systems (iSES 2026)*, NIT Goa, India, Dec. 2026.

Dr. B. R. Ambedkar National Institute of Technology Jalandhar, India.

## What this repository does

We train a lightweight transformer NIDS (FlowTransformer's 2-layer encoder with a Last Token head, 239,649 parameters on NSL-KDD and 679,749 on NF-UNSW-NB15-v2) several times with different seeds. We then explain every trained model (an **instance**) with KernelSHAP on the same 80 test flows and the same 40-flow background, and measure:

- **Comprehensiveness**: confidence drop when the top-5 SHAP features are masked;
- **Sufficiency gap**: confidence change when only the top-5 features are kept;
- **Stability**: cosine similarity between explanations before and after small Gaussian noise (σ = 0.03);
- **Noise floor**: cosine similarity between two explanations of the *unchanged* flow, i.e. KernelSHAP's own sampling noise.

Main observation: detection accuracy is nearly identical across instances, while comprehensiveness varies several times more (CV 13.8% vs 2.7% for F1 on NSL-KDD; 10.4% vs < 0.1% on NF-UNSW-NB15-v2), and stability sits at the noise floor.

No new detector or XAI method is proposed. The model is [FlowTransformer](https://github.com/liamdm/FlowTransformer) (Manocchio et al., *Expert Systems with Applications*, 2024), used unchanged apart from two compatibility patches for current pandas/Keras.

## Repository layout

```
.
├── framework/, implementations/   FlowTransformer (AGPL-3.0, by liamdm), two compatibility patches
├── prepare_nsl_kdd.py             builds data/NSL-KDD.csv from KDDTrain+.txt and KDDTest+.txt
├── prepare_nf_unsw.py             builds data/NF-UNSW-NB15-v2-prepared.csv from the parquet export
├── baseline_train.py              trains one NSL-KDD instance:            python baseline_train.py SEED
├── baseline_train_nf_unsw.py      trains one NF-UNSW-NB15-v2 instance:     python baseline_train_nf_unsw.py SEED
├── smoke_test.py                  10-step check that the pipeline runs
├── shap_utils.py                  windowed wrapper: SHAP perturbs only the explained flow, its 7-flow context stays fixed
├── test_shap_wrapper*.py          checks the wrapper agrees with FlowTransformer's own prediction
├── revision_eval_fo.py            the evaluation used for every result in the paper
├── edge_cost_benchmark.py         latency, throughput and SHAP cost on a CPU
├── make_tables_ieee.py            rebuilds Tables II-IV and the Fig. 1-2 coordinates from results/ (no TensorFlow needed)
├── feature_orders/                numeric input-column order each set of weights was trained with (see below)
├── results/
│   ├── detection/                 per-instance training records (F1, TP/FP/TN/FN, ...)
│   ├── shap_main/                 main runs: 80 flows, 500 samples, random background (Tables II-IV, Figs. 1-2)
│   ├── sigma_sweep/               20 flows re-explained at σ = 0.01/0.03/0.05/0.1 (estimator-noise analysis)
│   ├── bg_sensitivity/            20 flows with a class-stratified background (Sec. V-D)
│   ├── sample_budget/             150 and 1000 KernelSHAP samples on one instance (Sec. V-D)
│   ├── edge_cost/                 timing measurements on the laptop CPU (Sec. V-E)
│   └── derived/                   stats_v2.json (every statistic in the paper) and the regenerated tables
├── logs/                          console logs of the runs, including every VERIFY ... MATCH line
└── scripts/                       shell scripts exactly as they were run on the laptop (they assume the
                                   original flat working folder, with weights and records next to the code)
```

## Instances

The paper uses 9 NSL-KDD and 7 NF-UNSW-NB15-v2 instances. Labels: `dev` = original development environment, `sbx` = cloud sandbox, `lap`/`sig` = laptop (AMD Ryzen 5 5500U, CPU only). The number is the training seed.

| Dataset | Instance | Weights file | Detection record | Feature order |
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

The laptop and sandbox instances share one preprocessing cache, so they share one column order (`nsl_laptop.json`, `nfunsw_laptop.json`). The three `dev` instances were trained in their own processes; their orders were recovered from their saved outputs.

Result files are named `<dataset>_<instance>_ns<samples>_bg<mode><seed>[_sig<sigmas>].jsonl`, with one JSON line per explained flow (`row_index`, `true_label`, `pred_prob_attack`, `correct`, `saturated`, `top_features`, `comprehensiveness`, `sufficiency_gap`, `null_cos` (noise floor), `stability`, `all_feature_shap`, `seconds`). The matching `.meta.json` records the explained rows, the background rows, all settings and the library versions.

## Setup

Tested with Python 3.12, CPU only, on Ubuntu.

```bash
git clone https://github.com/priy4n5hu/shap-ft-nids.git
cd shap-ft-nids
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Weights.** Download `weights.zip` from the [Releases](../../releases) page and unzip it into the repository folder, so that the `.weights.h5` files are in `weights/`.

**Data.** The datasets are not redistributed here. Create a `data/` folder, then:

- **NSL-KDD**: download `KDDTrain+.txt` and `KDDTest+.txt` from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html) into `data/`, then run `python prepare_nsl_kdd.py`. The official train/test split is used.
- **NF-UNSW-NB15-v2** ([Sarhan et al., University of Queensland](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)): we used a cleaned parquet export with 1,986,745 flows. Run `python prepare_nf_unsw.py path/to/NF-UNSW-NB15-V2.parquet`. The script keeps FlowTransformer's 37 NetFlow-v2 features, caps each class (30,000 benign, up to 4,000 per attack category), makes a stratified 80/20 split and shuffles the rows (seed 42).
<!-- TODO before publishing: add the exact download link for the cleaned parquet export. -->

## Important: the input-column order

FlowTransformer builds its list of numeric input columns from a Python `set` (`framework/flow_transformer.py`, line 195), so their order depends on the process's hash seed. The first run writes a preprocessing cache to `cache/` and later runs reuse its order, but **a fresh clone builds a new, random order**. Keras loads weights without checking column names, so a model fed in the wrong order runs without any error and silently gives wrong predictions.

To prevent this, every command below passes the training order of that instance with `--feature-order`, and `--expect` recomputes the full test-set confusion matrix and aborts unless it matches the stored one. All 16 instances above print `VERIFY ... MATCH` from a fresh clone with no cache.

## Reproducing the results

Rebuild the paper's tables from the stored result files (seconds, no TensorFlow):

```bash
python make_tables_ieee.py
```

It prints Tables II-IV and the Fig. 1-2 coordinates, writes them to `results/derived/tables_ieee_corrected.tex`, and checks every value against `results/derived/stats_v2.json`.

Check one instance (under a minute):

```bash
python revision_eval_fo.py --dataset nsl --label lap47 \
    --weights weights/baseline_model_seed47.weights.h5 \
    --expect results/detection/baseline_results_seed47.json \
    --feature-order feature_orders/nsl_laptop.json --verify-only
```

Re-run the main explanation and evaluation for that instance (80 flows, 500 KernelSHAP samples; roughly 30 minutes for NSL-KDD and 70 for NF-UNSW-NB15-v2 on a laptop CPU). Use a new output folder: the script resumes from existing files in `--out-dir`.

```bash
python revision_eval_fo.py --dataset nsl --label lap47 \
    --weights weights/baseline_model_seed47.weights.h5 \
    --expect results/detection/baseline_results_seed47.json \
    --feature-order feature_orders/nsl_laptop.json --out-dir my_results
```

For NF-UNSW-NB15-v2 use `--dataset nfunsw` and the `nf_unsw_*` files. Other settings used in the paper:

| Analysis | Extra options |
|---|---|
| Main runs (Tables II-IV, Figs. 1-2) | none (defaults: `--n-per-class 40 --nsamples 500 --topk 5 --sigmas 0.03 --bg-mode random`) |
| σ sweep / estimator noise (Sec. V-C) | `--n-per-class 10 --sigmas 0.01,0.03,0.05,0.1` |
| Stratified background (Sec. V-D) | `--n-per-class 10 --bg-mode stratified` |
| Sample budget (Sec. V-D) | `--n-per-class 10 --nsamples 150` or `--nsamples 1000` |

KernelSHAP is stochastic, so a re-run gives values close to, but not bit-identical with, the stored files.

Cost on a CPU (Sec. V-E):

```bash
python edge_cost_benchmark.py --dataset nsl --label lap47 \
    --weights weights/baseline_model_seed47.weights.h5 \
    --expect results/detection/baseline_results_seed47.json \
    --feature-order feature_orders/nsl_laptop.json        # add --threads 1 for single-thread
```

Train a new instance (2-8 minutes on a laptop CPU):

```bash
python baseline_train.py 60          # NSL-KDD, seed 60
python baseline_train_nf_unsw.py 60  # NF-UNSW-NB15-v2, seed 60
```

Multi-threaded CPU training is not bit-reproducible even with a fixed seed, so a retrained model will differ slightly from the stored one. Models trained and evaluated in the same folder share `cache/` and need no `--feature-order`.

## License

AGPL-3.0, inherited from FlowTransformer (see `LICENSE`). The files in `framework/` and `implementations/` are by the FlowTransformer authors.

## Citation

If you use this code, please cite the paper above (see `CITATION.cff`) and FlowTransformer:

> L. D. Manocchio, S. Layeghy, W. W. Lo, G. K. Kulatilleke, M. Sarhan, and M. Portmann, "FlowTransformer: A transformer framework for flow-based network intrusion detection systems," *Expert Systems with Applications*, vol. 241, Art. no. 122564, 2024.
