"""
Converts the raw, headerless NSL-KDD KDDTrain+.txt / KDDTest+.txt files into
a single CSV matching framework.dataset_specification.NamedDatasetSpecifications.nsl_kdd:
  - 41 named feature columns (exact names FlowTransformer expects)
  - 'class' column (raw attack-name string; benign rows are 'normal')
  - 'is_test' column (0 = train split, 1 = official KDDTest+ split)
  - 'difficulty' column kept but NOT in include_fields, so FlowTransformer ignores it

Run once to produce data/NSL-KDD.csv, then point FlowTransformer's dataset_path at it.
"""
import pandas as pd

# Official NSL-KDD 41 feature names, in file-column order (KDD Cup 99 / NSL-KDD standard).
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'class', 'difficulty',
]
assert len(COLUMNS) == 43  # 41 features + class + difficulty

def load(path, is_test: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df['is_test'] = is_test
    return df

if __name__ == "__main__":
    train = load("data/KDDTrain+.txt", is_test=0)
    test = load("data/KDDTest+.txt", is_test=1)
    full = pd.concat([train, test], ignore_index=True)

    print(f"Train rows: {len(train)}  Test rows: {len(test)}  Total: {len(full)}")
    print(f"Attack classes present: {sorted(full['class'].unique())}")
    attack_rate = (full['class'] != 'normal').mean()
    print(f"Overall attack rate: {attack_rate:.3f}")

    full.to_csv("data/NSL-KDD.csv", index=False)
    print("Wrote data/NSL-KDD.csv")
