"""
Day 5: prepare NF-UNSW-NB15-v2 for FlowTransformer, from the real parquet
file (the "cleaned" V2 export -- 1,986,745 rows, 0 nulls, 0 duplicates,
no raw IPV4_SRC_ADDR/IPV4_DST_ADDR columns -- confirmed by inspection,
not assumed).

Field list and categorical/numeric split follow FlowTransformer's OWN
pre-built specs for this exact NetFlow-v2 feature family
(NamedDatasetSpecifications.unified_flow_format / cse_cic_ids_2018 in
framework/dataset_specification.py) rather than my own guess -- e.g. the
framework authors treat L4_SRC_PORT/L4_DST_PORT as categorical, which
I would have gotten wrong left to my own reasoning. Their reference list
excludes 4 fields present in this parquet (DNS_QUERY_ID, DNS_QUERY_TYPE,
DNS_TTL_ANSWER, FTP_COMMAND_RET_CODE) -- dropped here too, for consistency
with established precedent rather than introducing a new inconsistent
choice.

Heavily imbalanced dataset (Benign 1,911,666 / attack 75,079 total across
9 categories, from Exploits=29,905 down to Worms=164) -- samples per
class with a cap on the large classes, keeping ALL of the rare ones
(otherwise Worms/Backdoor/Analysis would nearly vanish).
"""
import numpy as np
import pandas as pd

from framework.dataset_specification import DatasetSpecification

SEED = 42
SAMPLE_PER_CLASS_CAP = 4000   # applied to each ATTACK category (9 of them)
BENIGN_CAP = 30000            # Benign is a single class but should dominate
                                # the working set the way it dominates the
                                # real population (96%) -- capping it at the
                                # same per-class number as each individual
                                # attack category (a real mistake in an
                                # earlier version of this script) inflated
                                # the attack rate to 82% and produced a
                                # trivially easy, unrealistic 99.99% F1.
TEST_FRAC = 0.2

# Same 37 fields as FlowTransformer's own unified_flow_format/cse_cic_ids_2018
# specs (established precedent for this exact NetFlow-v2 feature family).
INCLUDE_FIELDS = [
    'NUM_PKTS_UP_TO_128_BYTES', 'SRC_TO_DST_SECOND_BYTES', 'OUT_PKTS', 'OUT_BYTES',
    'NUM_PKTS_128_TO_256_BYTES', 'DST_TO_SRC_AVG_THROUGHPUT', 'DURATION_IN',
    'L4_SRC_PORT', 'ICMP_TYPE', 'PROTOCOL', 'SERVER_TCP_FLAGS', 'IN_PKTS',
    'NUM_PKTS_512_TO_1024_BYTES', 'CLIENT_TCP_FLAGS', 'TCP_WIN_MAX_IN',
    'NUM_PKTS_256_TO_512_BYTES', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN',
    'LONGEST_FLOW_PKT', 'L4_DST_PORT', 'MIN_TTL', 'DST_TO_SRC_SECOND_BYTES',
    'NUM_PKTS_1024_TO_1514_BYTES', 'DURATION_OUT', 'FLOW_DURATION_MILLISECONDS',
    'TCP_FLAGS', 'MAX_TTL', 'SRC_TO_DST_AVG_THROUGHPUT', 'ICMP_IPV4_TYPE',
    'MAX_IP_PKT_LEN', 'RETRANSMITTED_OUT_BYTES', 'IN_BYTES', 'RETRANSMITTED_IN_BYTES',
    'TCP_WIN_MAX_OUT', 'L7_PROTO', 'RETRANSMITTED_OUT_PKTS', 'RETRANSMITTED_IN_PKTS',
]
CATEGORICAL_FIELDS = [
    'CLIENT_TCP_FLAGS', 'L4_SRC_PORT', 'TCP_FLAGS', 'ICMP_IPV4_TYPE',
    'ICMP_TYPE', 'PROTOCOL', 'SERVER_TCP_FLAGS', 'L4_DST_PORT', 'L7_PROTO',
]

nf_unsw_nb15_v2 = DatasetSpecification(
    include_fields=INCLUDE_FIELDS,
    categorical_fields=CATEGORICAL_FIELDS,
    class_column="Attack",
    benign_label="Benign",
    test_column="is_test",
)

if __name__ == "__main__":
    print("Reading parquet...")
    import sys
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else "data/NF-UNSW-NB15-V2.parquet"
    df = pd.read_parquet(parquet_path)
    print(f"Full shape: {df.shape}")

    missing = [c for c in INCLUDE_FIELDS if c not in df.columns]
    if missing:
        raise SystemExit(f"STOPPING -- expected columns not found: {missing}")
    print("All 37 expected feature columns present. Proceeding.")

    rng = np.random.RandomState(SEED)
    parts = []
    for cls, group in df.groupby("Attack"):
        cap = BENIGN_CAP if cls == "Benign" else SAMPLE_PER_CLASS_CAP
        n = min(len(group), cap)
        parts.append(group.sample(n=n, random_state=rng))
    sampled = pd.concat(parts, ignore_index=True)
    print(f"\nAfter per-class cap (Benign={BENIGN_CAP}, attack classes={SAMPLE_PER_CLASS_CAP}):")
    print(sampled["Attack"].value_counts())
    attack_rate = (sampled["Attack"] != "Benign").mean()
    print(f"Total: {len(sampled)}  (attack rate: {attack_rate:.1%}, "
          f"vs. {(df['Attack']!='Benign').mean():.1%} in the full population)")

    # Stratified train/test split (no official split exists for this dataset).
    is_test = np.zeros(len(sampled), dtype=int)
    for cls, group in sampled.groupby("Attack"):
        idx = group.index.values
        n_test = max(1, int(round(len(idx) * TEST_FRAC)))
        test_idx = rng.choice(idx, size=n_test, replace=False)
        is_test[test_idx] = 1
    sampled["is_test"] = is_test
    print(f"\nTrain/test split: {(sampled['is_test']==0).sum()} train, "
          f"{(sampled['is_test']==1).sum()} test")

    # CRITICAL: shuffle row order before writing. groupby+concat above left
    # rows clustered in contiguous same-Attack-class blocks (confirmed:
    # 100% of consecutive rows shared the same label pre-shuffle). Since
    # FlowTransformer builds each flow's context window from the preceding
    # rows in FILE order, unshuffled data would give every window 7
    # same-class neighbors -- near-free signal via context homogeneity,
    # not genuine sequence modeling. Caught this via a direct check after
    # the first training run came back at a suspicious, exactly-perfect
    # F1=1.0 -- fixed here rather than reporting that number.
    sampled = sampled.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    same_as_prev = (sampled["Attack"] == sampled["Attack"].shift()).mean()
    print(f"After shuffling: fraction of consecutive rows sharing the same "
          f"Attack label = {same_as_prev:.3f} (was 1.000 before the fix)")

    out_cols = INCLUDE_FIELDS + ["Attack", "is_test"]
    import os
    os.makedirs("data", exist_ok=True)
    out_path = "data/NF-UNSW-NB15-v2-prepared.csv"
    sampled[out_cols].to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  (shape: {sampled[out_cols].shape})")
