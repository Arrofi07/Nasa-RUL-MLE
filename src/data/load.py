"""
Load & preprocess NASA Turbofan (RUL) dataset.

Key change from v1
------------------
The RUL cap is now a parameter called `rul_cap` (default 125) instead of a
hardcoded constant. This lets you run the full pipeline with different cap
values (e.g. 125, 130, 135, 140) and compare test RMSE in MLflow to find
the best value. See the module docstring in tune_xgb_optuna_mlflow.py for
how to wire this into the training scripts.

Why we clip RUL at all
----------------------
In the early cycles of an engine's life the sensors look nearly identical
regardless of whether the engine has 150 cycles left or 300 cycles left —
the degradation signal hasn't started yet. Predicting "RUL = 280 vs 150"
from sensor noise would be a guess that adds variance without adding
information. The cap says: "we only trust predictions within the last
`rul_cap` cycles, so treat anything beyond that as equally healthy."

Why 125 might be slightly too low for FD001
--------------------------------------------
The test set contains engines with true RUL up to ~145 cycles. With a cap
of 125 the model has never seen a training target above 125, so it can never
predict above ~115 in practice. You see this as the hard ceiling in the
Predicted vs True scatter plot. A cap of 130–135 gives the model room to
express higher confidence for genuinely early-life engines without losing
the regularisation benefit.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")

# The default cap — can be overridden from any training script.
# 125 is the standard FD001 value used in most published papers.
DEFAULT_RUL_CAP = 125


def load_and_preprocess_data(
    train_path: str = "data/raw/train_FD001.txt",
    test_path: str = "data/raw/test_FD001.txt",
    rul_path: str = "data/raw/RUL_FD001.txt",
    output_dir: Path | str = DATA_DIR,
    clip_rul: bool = True,
    rul_cap: int = DEFAULT_RUL_CAP,  # ← NEW: was hardcoded to 125
):
    """
    Load raw NASA FD001 files, compute RUL labels, and save processed CSVs.

    Parameters
    ----------
    train_path, test_path, rul_path
        Paths to the three raw NASA text files.
    output_dir
        Directory to write train.csv, test.csv, rul.csv.
        Tests can pass a temp dir so nothing in data/ is touched.
    clip_rul : bool
        Whether to apply the RUL cap at all.
        Set False only for ablation experiments — the model loses its
        regularisation anchor and will likely overfit early-life engines.
    rul_cap : int
        Maximum RUL value in the training labels (cycles).
        Any engine cycle with more than `rul_cap` cycles remaining is
        treated as if it has exactly `rul_cap` cycles left.
        Recommended search range for FD001: 125, 130, 135, 140.
    """

    # -----------------------------------------------------------------------
    # Load raw text files (space-separated, no header)
    # -----------------------------------------------------------------------
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None)
    test_df  = pd.read_csv(test_path,  sep=r"\s+", header=None)
    rul_df   = pd.read_csv(rul_path,   sep=r"\s+", header=None)

    # Assign column names: engine_id, cycle, 3 settings, 21 sensors
    cols_name = (
        ["engine_id", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}"  for i in range(1, 22)]
    )
    train_df.columns = cols_name
    test_df.columns  = cols_name
    rul_df.columns   = ["rul"]

    # -----------------------------------------------------------------------
    # Compute RUL labels for the training set
    # RUL at cycle t = (max cycle for this engine) − t
    # -----------------------------------------------------------------------
    max_cycle = train_df.groupby("engine_id")["cycle"].max()

    train_df["rul"] = train_df.apply(
        lambda row: max_cycle[row.engine_id] - row.cycle,
        axis=1,
    )

    # -----------------------------------------------------------------------
    # Apply RUL cap (the key change)
    # -----------------------------------------------------------------------
    if clip_rul:
        # All rows where rul > rul_cap are set to rul_cap.
        # This flattens the target for healthy early-life engines so the model
        # doesn't try to distinguish "350 cycles left" from "200 cycles left"
        # when the sensors can't support that distinction anyway.
        train_df["rul"] = train_df["rul"].clip(upper=rul_cap)
        print(f"✅ RUL clipped at {rul_cap} cycles (training set only).")
    else:
        print("⚠️  RUL clipping disabled — ablation mode.")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(outdir / "train.csv", index=False)
    test_df.to_csv(outdir / "test.csv",  index=False)
    rul_df.to_csv(outdir / "rul.csv",   index=False)

    print(f"✅ Data saved to {outdir}")
    print(f"   Train : {train_df.shape}  |  RUL range: {train_df['rul'].min():.0f}–{train_df['rul'].max():.0f}")
    print(f"   Test  : {test_df.shape}")
    print(f"   RUL   : {rul_df.shape}    |  True RUL range: {rul_df['rul'].min():.0f}–{rul_df['rul'].max():.0f}")

    return train_df, test_df, rul_df


if __name__ == "__main__":
    # Default run: standard 125-cycle cap
    load_and_preprocess_data()