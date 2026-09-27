# Dataset Documentation: UNSW-NB15

This directory contains the training and evaluation datasets derived from the **UNSW-NB15 Network Intrusion Dataset** created by the Australian Centre for Cyber Security (ACCS).

## Files Included in Git

- `UNSW_NB15_training-set.csv` (~14.7 MB, 82,332 records): Configured training set partition across 10 attack categories and normal traffic.
- `UNSW_NB15_testing-set.csv` (~30.8 MB, 175,341 records): Evaluation test set partition.
- `NUSW-NB15_features.csv` (4 KB): Specification of 49 total extracted network features and their data types.
- `UNSW-NB15_LIST_EVENTS.csv` (4 KB): Event and attack timeline logs.
- `train_test_network.csv` (~28.5 MB): Consolidated network flow features.

## Raw Large Files (Ignored via `.gitignore`)

Due to GitHub's 100 MB per-file limit, the following uncompressed raw CSVs are placed in `data/raw/` and excluded from git tracking:
- `UNSW-NB15_1.csv` (~161 MB)
- `UNSW-NB15_2.csv` (~158 MB)
- `UNSW-NB15_3.csv` (~147 MB)
- `UNSW-NB15_4.csv` (~93 MB)

If you require the complete uncompressed packet captures or raw multi-gigabyte CSV records, they can be downloaded from the official source:
- [UNSW-NB15 Dataset Official Repository (UNSW Canberra / ACCS)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
