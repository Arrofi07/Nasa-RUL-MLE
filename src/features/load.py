"""
Load & preprocess NASA Turbofan (RUL) dataset.

- Production default writes to data/raw/
- Tests can pass a temp `output_dir` so nothing in data/ is touched.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")


def load_and_preprocess_data(
    train_path: str = "data/raw/train_FD001.txt",
    test_path: str = "data/raw/test_FD001.txt",
    rul_path: str = "data/raw/RUL_FD001.txt",
    output_dir: Path | str = DATA_DIR,
    clip_rul: bool = True,
):
    """Load raw NASA dataset, preprocess (compute RUL), and save to output_dir."""

    # Load raw data
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None)

    # Column names
    cols_name = (
        ["engine_id", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    # Assign column names
    train_df.columns = cols_name
    test_df.columns = cols_name
    rul_df.columns = ["rul"]

    # Compute RUL for train
    max_cycle = train_df.groupby("engine_id")["cycle"].max()

    train_df["rul"] = train_df.apply(
        lambda row: max_cycle[row.engine_id] - row.cycle,
        axis=1,
    )

    # Optional clipping (common trick for turbofan dataset)
    if clip_rul:
        train_df["rul"] = train_df["rul"].clip(upper=125)

    # Save outputs
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(outdir / "train.csv", index=False)
    test_df.to_csv(outdir / "test.csv", index=False)
    rul_df.to_csv(outdir / "rul.csv", index=False)

    print(f"✅ Data preprocessing completed (saved to {outdir}).")
    print(f"   Train: {train_df.shape}")
    print(f"   Test: {test_df.shape}")
    print(f"   RUL: {rul_df.shape}")

    return train_df, test_df, rul_df


if __name__ == "__main__":
    load_and_preprocess_data()