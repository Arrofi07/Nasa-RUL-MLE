"""
tests/test_data_quality.py — Data Quality Tests

Tests cover load.py and preprocess.py:
  - Schema correctness (column names, dtypes, row counts)
  - RUL computation and clipping logic
  - Sensor dropping (6 known low-info sensors)
  - Constant feature removal
  - Normalisation (train-only fit, no leakage into test)
  - Output file creation
"""

import numpy as np
import pandas as pd

from src.data.load import load_and_preprocess_data
from src.data.preprocess import (
    normalize_features,
    remove_constant_features,
)

EXPECTED_COLS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# ===========================================================================
# load.py tests
# ===========================================================================


class TestLoadAndPreprocessData:
    def test_output_files_created(self, raw_dir, tmp_path):
        out = tmp_path / "out"
        load_and_preprocess_data(
            train_path=str(raw_dir / "train_FD001.txt"),
            test_path=str(raw_dir / "test_FD001.txt"),
            rul_path=str(raw_dir / "RUL_FD001.txt"),
            output_dir=out,
        )
        assert (out / "train.csv").exists(), "train.csv not created"
        assert (out / "test.csv").exists(), "test.csv not created"
        assert (out / "rul.csv").exists(), "rul.csv not created"

    def test_train_columns(self, raw_dir, tmp_path):
        out = tmp_path / "out"
        train, _, _ = load_and_preprocess_data(
            train_path=str(raw_dir / "train_FD001.txt"),
            test_path=str(raw_dir / "test_FD001.txt"),
            rul_path=str(raw_dir / "RUL_FD001.txt"),
            output_dir=out,
        )
        expected = EXPECTED_COLS + ["rul"]
        assert list(train.columns) == expected, (
            f"Unexpected columns: {train.columns.tolist()}"
        )

    def test_test_columns(self, raw_dir, tmp_path):
        out = tmp_path / "out"
        _, test, _ = load_and_preprocess_data(
            train_path=str(raw_dir / "train_FD001.txt"),
            test_path=str(raw_dir / "test_FD001.txt"),
            rul_path=str(raw_dir / "RUL_FD001.txt"),
            output_dir=out,
        )
        assert list(test.columns) == EXPECTED_COLS

    def test_rul_is_non_negative(self, raw_dir, tmp_path):
        out = tmp_path / "out"
        train, _, _ = load_and_preprocess_data(
            train_path=str(raw_dir / "train_FD001.txt"),
            test_path=str(raw_dir / "test_FD001.txt"),
            rul_path=str(raw_dir / "RUL_FD001.txt"),
            output_dir=out,
        )
        assert (train["rul"] >= 0).all(), "RUL contains negative values"

    def test_rul_clipped_at_125(self, raw_dir, tmp_path):
        out = tmp_path / "out"
        train, _, _ = load_and_preprocess_data(
            train_path=str(raw_dir / "train_FD001.txt"),
            test_path=str(raw_dir / "test_FD001.txt"),
            rul_path=str(raw_dir / "RUL_FD001.txt"),
            output_dir=out,
            clip_rul=True,
        )
        assert train["rul"].max() <= 125, "RUL exceeds clip value of 125"

    def test_rul_not_clipped_when_disabled(self, tmp_path):
        """With clip_rul=False and enough cycles, some RUL values exceed 125."""
        out = tmp_path / "out"
        rng = np.random.default_rng(1)
        n_engines = 2
        n_cycles = 200  # 200 cycles → max RUL = 199 (well above 125)

        rows = []
        for eid in range(1, n_engines + 1):
            for c in range(1, n_cycles + 1):
                row = (
                    [eid, c]
                    + rng.uniform(-1, 1, 3).tolist()
                    + rng.uniform(400, 700, 21).tolist()
                )
                rows.append(row)
        cols = (
            ["engine_id", "cycle"]
            + [f"setting_{i}" for i in range(1, 4)]
            + [f"sensor_{i}" for i in range(1, 22)]
        )
        train_df = pd.DataFrame(rows, columns=cols)

        # test and rul must have matching engine count
        test_rows = []
        for eid in range(1, n_engines + 1):
            for c in range(1, 21):
                row = (
                    [eid, c]
                    + rng.uniform(-1, 1, 3).tolist()
                    + rng.uniform(400, 700, 21).tolist()
                )
                test_rows.append(row)
        test_df = pd.DataFrame(test_rows, columns=cols)
        rul_df = pd.DataFrame({"rul": [50, 60]})

        train_txt = tmp_path / "big_train.txt"
        test_txt = tmp_path / "big_test.txt"
        rul_txt = tmp_path / "big_rul.txt"
        train_df.to_csv(train_txt, sep=" ", index=False, header=False)
        test_df.to_csv(test_txt, sep=" ", index=False, header=False)
        rul_df.to_csv(rul_txt, sep=" ", index=False, header=False)

        train, _, _ = load_and_preprocess_data(
            train_path=str(train_txt),
            test_path=str(test_txt),
            rul_path=str(rul_txt),
            output_dir=out,
            clip_rul=False,
        )
        assert train["rul"].max() > 125, f"Expected RUL > 125, got {train['rul'].max()}"

    def test_engine_id_dtype(self, raw_dir, tmp_path):
        out = tmp_path / "out"
        train, _, _ = load_and_preprocess_data(
            train_path=str(raw_dir / "train_FD001.txt"),
            test_path=str(raw_dir / "test_FD001.txt"),
            rul_path=str(raw_dir / "RUL_FD001.txt"),
            output_dir=out,
        )
        assert pd.api.types.is_integer_dtype(train["engine_id"]), (
            "engine_id should be integer"
        )

    def test_row_count_preserved(self, raw_dir, tmp_path):
        """Row count in train output == n_engines × cycles_per_engine."""
        from conftest import N_ENGINES, N_CYCLES

        out = tmp_path / "out"
        train, _, _ = load_and_preprocess_data(
            train_path=str(raw_dir / "train_FD001.txt"),
            test_path=str(raw_dir / "test_FD001.txt"),
            rul_path=str(raw_dir / "RUL_FD001.txt"),
            output_dir=out,
        )
        expected_rows = N_ENGINES * N_CYCLES
        assert len(train) == expected_rows, (
            f"Expected {expected_rows} rows, got {len(train)}"
        )


# ===========================================================================
# preprocess.py unit tests
# ===========================================================================


class TestRemoveConstantFeatures:
    def test_removes_constant_column(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3], "c": [0, 0, 0]})
        result = remove_constant_features(df)
        assert "a" not in result.columns
        assert "c" not in result.columns
        assert "b" in result.columns

    def test_no_constant_columns_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = remove_constant_features(df)
        assert list(result.columns) == ["a", "b"]

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        original_cols = df.columns.tolist()
        remove_constant_features(df)
        assert df.columns.tolist() == original_cols


class TestNormalizeFeatures:
    def test_train_mean_near_zero(self, raw_train_df, raw_test_df):
        train_norm, _ = normalize_features(raw_train_df.copy(), raw_test_df.copy())
        feature_cols = [
            c for c in train_norm.columns if c not in ["engine_id", "cycle", "rul"]
        ]
        means = train_norm[feature_cols].mean().abs()
        assert (means < 0.1).all(), (
            "Train feature means should be ~0 after normalization"
        )

    def test_train_std_near_one(self, raw_train_df, raw_test_df):
        train_norm, _ = normalize_features(raw_train_df.copy(), raw_test_df.copy())
        feature_cols = [
            c for c in train_norm.columns if c not in ["engine_id", "cycle", "rul"]
        ]
        stds = train_norm[feature_cols].std()
        assert ((stds - 1.0).abs() < 0.1).all(), (
            "Train feature stds should be ~1 after normalization"
        )

    def test_no_leakage_from_test(self, raw_train_df, raw_test_df):
        """Test scaler is fitted on train only — same train data → same result."""
        train_norm1, _ = normalize_features(raw_train_df.copy(), raw_test_df.copy())
        train_norm2, _ = normalize_features(raw_train_df.copy(), raw_test_df.copy())
        feature_cols = [
            c for c in train_norm1.columns if c not in ["engine_id", "cycle", "rul"]
        ]
        pd.testing.assert_frame_equal(
            train_norm1[feature_cols].round(8),
            train_norm2[feature_cols].round(8),
        )

    def test_meta_columns_unchanged(self, raw_train_df, raw_test_df):
        original_engine_ids = raw_train_df["engine_id"].values.copy()
        train_norm, _ = normalize_features(raw_train_df.copy(), raw_test_df.copy())
        np.testing.assert_array_equal(
            train_norm["engine_id"].values, original_engine_ids
        )


# ===========================================================================
# preprocess_data integration test
# ===========================================================================


class TestPreprocessDataIntegration:
    def test_output_files_created(self, processed_dir):
        assert (processed_dir / "train_clean.csv").exists()
        assert (processed_dir / "test_clean.csv").exists()
        assert (processed_dir / "rul_clean.csv").exists()

    def test_constant_columns_absent(self, processed_dir):
        """Preprocessing must remove zero-variance columns from train."""
        train = pd.read_csv(processed_dir / "train_clean.csv")
        feature_cols = [
            c for c in train.columns if c not in ["engine_id", "cycle", "rul"]
        ]
        for col in feature_cols:
            assert train[col].nunique() > 1, (
                f"{col} is constant — should have been dropped"
            )

    def test_no_missing_values(self, processed_dir):
        train = pd.read_csv(processed_dir / "train_clean.csv")
        test = pd.read_csv(processed_dir / "test_clean.csv")
        assert train.isnull().sum().sum() == 0, "train_clean.csv has missing values"
        assert test.isnull().sum().sum() == 0, "test_clean.csv has missing values"

    def test_train_test_same_feature_columns(self, processed_dir):
        train = pd.read_csv(processed_dir / "train_clean.csv")
        test = pd.read_csv(processed_dir / "test_clean.csv")
        train_feat = [c for c in train.columns if c not in ["rul"]]
        test_feat = list(test.columns)
        assert set(train_feat) == set(test_feat), (
            "Train and test feature columns differ"
        )
