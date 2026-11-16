# Data/ — raw inputs (not tracked by git)

Everything in this folder except this README is ignored by git (too large for GitHub).
Download the files from the shared Google Drive folder **`final_project`** (owner: Debbie) and put them
here, flat, with the same names. The notebook's setup cell looks for `Data/` relative to the repo root.

| File | Size | What it is | Needed for |
|---|---|---|---|
| `datashare_small.parquet` | 563 MB | Gu–Kelly–Xiu (GKX) monthly firm-characteristic panel, 1980+ subset already in parquet | **everything** (what the notebook actually reads) |
| `datashare.zip` / `datashare.csv` | 1.45 GB / ~11 GB | the full raw GKX panel the parquet was made from | only to rebuild the parquet |
| `datashare_sample.parquet` | 14 MB | tiny sample of the panel for quick tests | optional |
| `Data2024.xlsx - Monthly.csv` | 510 KB | Goyal–Welch monthly predictor data | §1, §6 |
| `Data2025.xlsx - Quarterly.csv`, `Data2025.xlsx - Annual.csv` | 181 KB, 49 KB | Goyal–Welch quarterly / annual series | §6 (optional block, skipped if absent) |
| `F-F_Research_Data_5_Factors_2x3_CSV.zip` | 12 KB | Fama–French 5 factors (monthly) | §1, §6 |

Outputs the notebook writes (`panel_top500_clean.parquet`, `panel_top500_features.parquet`, …) go to
`data_clean/` at the repo root, also ignored by git. The small CSV tables it writes to
`Data Definition and Summary Statistics/` (data dictionary, summary stats, leakage audit) are fine to commit.

On Colab none of this is needed: the setup cell mounts Drive and reads the same files from
`MyDrive/final_project` (add the shared folder as a shortcut to My Drive).
