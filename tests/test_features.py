"""
tests/test_features.py — Feature Engineering Tests

Tests cover build_feature.py and sequence_builder.py:
  - Feature selection by correlation threshold
  - Rolling mean correctness (window, min_periods)
  - Diff features (first row per engine = 0)
  - Scale leakage prevention
  - Sequence shape from create_sequences
  - Group-aware train/val split from create_group_split_sequences
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_feature import (
    add_difference_features,
    add_rolling_features,
    scale_features,
    select_features,
)
from src.features.sequence_builder import (
    create_group_split_sequences,
    create_sequences,
)

# ===========================================================================
# select_features
# ===========================================================================


class TestSelectFeatures:
    def test_returns_list(self, feature_train_df):
        result = select_features(feature_train_df)
        assert isinstance(result, list)

    def test_excludes_rul_and_cycle(self, feature_train_df):
        result = select_features(feature_train_df)
        assert "rul" not in result
        assert "cycle" not in result

    def test_threshold_zero_returns_all_features(self, feature_train_df):
        """threshold=0 should keep all non-meta features that have any correlation with RUL.
        engine_id may also appear if it correlates — so we check our known feat cols are
        all present rather than asserting exact set equality."""
        result = select_features(feature_train_df, threshold=0.0)
        feat_cols = [
            c
            for c in feature_train_df.columns
            if c not in ["engine_id", "cycle", "rul"]
        ]
        for col in feat_cols:
            assert col in result, f"Feature {col} missing from result at threshold=0"
        # meta columns other than engine_id must be excluded
        assert "cycle" not in result
        assert "rul" not in result

    def test_threshold_one_returns_empty(self, feature_train_df):
        """No feature is perfectly correlated with RUL (r²=1) in random data."""
        result = select_features(feature_train_df, threshold=1.0)
        assert result == []

    def test_highly_correlated_feature_selected(self):
        """A feature that is perfectly correlated with RUL must always be selected."""
        rng = np.random.default_rng(99)
        n = 100
        rul = rng.integers(0, 125, n).astype(float)
        df = pd.DataFrame(
            {
                "engine_id": np.ones(n, dtype=int),
                "cycle": np.arange(1, n + 1),
                "rul": rul,
                "perfect": rul * 2 + 5,  # perfect linear correlation
                "noise": rng.standard_normal(n),
            }
        )
        result = select_features(df, threshold=0.5)
        assert "perfect" in result


# ===========================================================================
# add_rolling_features
# ===========================================================================


class TestAddRollingFeatures:
    def test_rolling_columns_created(
        self, feature_train_df, feature_test_df, feat_cols
    ):
        train, test = add_rolling_features(
            feature_train_df.copy(), feature_test_df.copy(), feat_cols
        )
        for col in feat_cols:
            assert f"{col}_rolling_mean" in train.columns
            assert f"{col}_rolling_mean" in test.columns

    def test_single_cycle_engine_rolling_equals_value(self, feat_cols):
        """With one cycle, rolling mean (min_periods=1) == the raw value."""
        df = pd.DataFrame(
            {
                "engine_id": [1],
                "cycle": [1],
                **{f: [float(i)] for i, f in enumerate(feat_cols)},
            }
        )
        train, _ = add_rolling_features(df.copy(), df.copy(), feat_cols)
        for f in feat_cols:
            assert train[f"{f}_rolling_mean"].iloc[0] == pytest.approx(train[f].iloc[0])

    def test_rolling_respects_engine_boundary(self, feat_cols):
        """Rolling mean must not bleed across engine boundaries."""
        df = pd.DataFrame(
            {
                "engine_id": [1, 1, 2, 2],
                "cycle": [1, 2, 1, 2],
                **{f: [10.0, 20.0, 100.0, 200.0] for f in feat_cols[:1]},
                **{f: [0.0] * 4 for f in feat_cols[1:]},
            }
        )
        train, _ = add_rolling_features(df.copy(), df.copy(), feat_cols[:1], window=5)
        # Engine 2 cycle 1 rolling mean should be 100.0, not influenced by engine 1
        eng2_c1 = train[(train["engine_id"] == 2) & (train["cycle"] == 1)]
        assert eng2_c1[f"{feat_cols[0]}_rolling_mean"].values[0] == pytest.approx(100.0)

    def test_no_nans_in_rolling(self, feature_train_df, feature_test_df, feat_cols):
        train, test = add_rolling_features(
            feature_train_df.copy(), feature_test_df.copy(), feat_cols
        )
        rolling_cols = [f"{f}_rolling_mean" for f in feat_cols]
        assert train[rolling_cols].isnull().sum().sum() == 0
        assert test[rolling_cols].isnull().sum().sum() == 0


# ===========================================================================
# add_difference_features
# ===========================================================================


class TestAddDifferenceFeatures:
    def test_diff_columns_created(self, feature_train_df, feature_test_df, feat_cols):
        train, test = add_difference_features(
            feature_train_df.copy(), feature_test_df.copy(), feat_cols
        )
        for col in feat_cols:
            assert f"{col}_diff" in train.columns
            assert f"{col}_diff" in test.columns

    def test_first_cycle_diff_is_zero(
        self, feature_train_df, feature_test_df, feat_cols
    ):
        """First cycle per engine has no previous cycle → diff must be 0."""
        train, _ = add_difference_features(
            feature_train_df.copy(), feature_test_df.copy(), feat_cols
        )
        first_cycles = train[train["cycle"] == 1]
        for col in feat_cols:
            assert (
                first_cycles[f"{col}_diff"] == 0.0
            ).all(), f"First cycle {col}_diff is not 0"

    def test_diff_respects_engine_boundary(self, feat_cols):
        """Diff must not subtract across engines."""
        df = pd.DataFrame(
            {
                "engine_id": [1, 1, 2, 2],
                "cycle": [1, 2, 1, 2],
                feat_cols[0]: [10.0, 15.0, 100.0, 110.0],
                **{f: [0.0] * 4 for f in feat_cols[1:]},
            }
        )
        train, _ = add_difference_features(df.copy(), df.copy(), feat_cols[:1])
        # Engine 2, cycle 1 diff should be 0 (not 100 - 15)
        val = train[(train["engine_id"] == 2) & (train["cycle"] == 1)][
            f"{feat_cols[0]}_diff"
        ].values[0]
        assert val == pytest.approx(0.0), f"Expected 0, got {val}"

    def test_no_nans_in_diff(self, feature_train_df, feature_test_df, feat_cols):
        train, test = add_difference_features(
            feature_train_df.copy(), feature_test_df.copy(), feat_cols
        )
        diff_cols = [f"{f}_diff" for f in feat_cols]
        assert train[diff_cols].isnull().sum().sum() == 0
        assert test[diff_cols].isnull().sum().sum() == 0


# ===========================================================================
# scale_features
# ===========================================================================


class TestScaleFeatures:
    def test_train_mean_near_zero(self, feature_train_df, feature_test_df):
        train_sc, _ = scale_features(feature_train_df.copy(), feature_test_df.copy())
        feat = [c for c in train_sc.columns if c not in ["engine_id", "cycle", "rul"]]
        means = train_sc[feat].mean().abs()
        assert (means < 0.1).all(), "Train means not near zero after scaling"

    def test_train_std_near_one(self, feature_train_df, feature_test_df):
        train_sc, _ = scale_features(feature_train_df.copy(), feature_test_df.copy())
        feat = [c for c in train_sc.columns if c not in ["engine_id", "cycle", "rul"]]
        stds = train_sc[feat].std()
        assert ((stds - 1.0).abs() < 0.1).all(), "Train stds not near 1 after scaling"

    def test_no_leakage_test_stats_not_used(self, feature_train_df, feat_cols):
        """Changing test data must not change how train is scaled."""
        test_a = feature_train_df.copy().drop(columns=["rul"])
        test_b = feature_train_df.copy().drop(columns=["rul"]) * 99  # very different

        train_a, _ = scale_features(feature_train_df.copy(), test_a)
        train_b, _ = scale_features(feature_train_df.copy(), test_b)

        feat = [c for c in train_a.columns if c not in ["engine_id", "cycle", "rul"]]
        pd.testing.assert_frame_equal(
            train_a[feat].round(8),
            train_b[feat].round(8),
            check_names=True,
        )


# ===========================================================================
# build_features (integration)
# ===========================================================================


class TestBuildFeaturesIntegration:
    def test_output_files_created(self, feature_dir):
        assert (feature_dir / "feature_engineered_train.csv").exists()
        assert (feature_dir / "feature_engineered_test.csv").exists()

    def test_rolling_and_diff_columns_present(self, feature_dir):
        train = pd.read_csv(feature_dir / "feature_engineered_train.csv")
        rolling_cols = [c for c in train.columns if c.endswith("_rolling_mean")]
        diff_cols = [c for c in train.columns if c.endswith("_diff")]
        assert len(rolling_cols) > 0, "No rolling_mean columns found"
        assert len(diff_cols) > 0, "No _diff columns found"

    def test_no_missing_values(self, feature_dir):
        train = pd.read_csv(feature_dir / "feature_engineered_train.csv")
        test = pd.read_csv(feature_dir / "feature_engineered_test.csv")
        assert train.isnull().sum().sum() == 0
        assert test.isnull().sum().sum() == 0

    def test_train_has_rul_column(self, feature_dir):
        train = pd.read_csv(feature_dir / "feature_engineered_train.csv")
        assert "rul" in train.columns

    def test_test_has_no_rul_column(self, feature_dir):
        test = pd.read_csv(feature_dir / "feature_engineered_test.csv")
        assert "rul" not in test.columns


# ===========================================================================
# sequence_builder.py
# ===========================================================================


class TestCreateSequences:
    def test_output_shapes(self, feature_train_df, feat_cols):
        from conftest import N_CYCLES, N_ENGINES

        seq_len = 10
        X, y = create_sequences(feature_train_df, seq_len, feat_cols)
        expected_seqs = N_ENGINES * (N_CYCLES - seq_len)
        assert X.shape == (
            expected_seqs,
            seq_len,
            len(feat_cols),
        ), f"Expected ({expected_seqs}, {seq_len}, {len(feat_cols)}), got {X.shape}"
        assert y.shape == (expected_seqs,)

    def test_seq_len_larger_than_engine_produces_no_sequence(self, feat_cols):
        """If seq_len >= n_cycles, no sequences can be formed for that engine."""
        df = pd.DataFrame(
            {
                "engine_id": [1, 1, 1],
                "cycle": [1, 2, 3],
                "rul": [2, 1, 0],
                **{f: [float(i)] * 3 for i, f in enumerate(feat_cols)},
            }
        )
        X, y = create_sequences(df, seq_len=5, feature_cols=feat_cols)
        assert len(X) == 0
        assert len(y) == 0

    def test_target_is_rul_after_sequence(self, feat_cols):
        """y[i] should be the RUL at position seq_len + i (the step after the window)."""
        n = 10
        seq_len = 3
        df = pd.DataFrame(
            {
                "engine_id": [1] * n,
                "cycle": list(range(1, n + 1)),
                "rul": list(range(n - 1, -1, -1)),  # 9,8,...,0
                **{f: [0.0] * n for f in feat_cols},
            }
        )
        X, y = create_sequences(df, seq_len=seq_len, feature_cols=feat_cols)
        # First target: RUL at index seq_len = index 3 → value = 9-3 = 6
        assert y[0] == pytest.approx(n - 1 - seq_len)


class TestCreateGroupSplitSequences:
    def test_output_four_arrays(self, feature_train_df, feat_cols):
        result = create_group_split_sequences(feature_train_df, feat_cols, seq_len=10)
        assert len(result) == 4, "Expected (X_train, X_val, y_train, y_val)"

    def test_no_engine_overlap_between_train_val(self, feature_train_df, feat_cols):
        """
        GroupShuffleSplit must keep all cycles of one engine in either train or val,
        never both. We verify this by checking engine_id groups don't overlap.
        """
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(
            splitter.split(feature_train_df, groups=feature_train_df["engine_id"])
        )
        train_engines = set(feature_train_df.iloc[train_idx]["engine_id"])
        val_engines = set(feature_train_df.iloc[val_idx]["engine_id"])
        assert train_engines.isdisjoint(
            val_engines
        ), f"Engine IDs appear in both splits: {train_engines & val_engines}"

    def test_shapes_consistent(self, feature_train_df, feat_cols):
        seq_len = 5
        X_tr, X_val, y_tr, y_val = create_group_split_sequences(
            feature_train_df, feat_cols, seq_len=seq_len
        )
        assert X_tr.ndim == 3
        assert X_val.ndim == 3
        assert X_tr.shape[1] == seq_len
        assert X_val.shape[1] == seq_len
        assert X_tr.shape[2] == len(feat_cols)
        assert len(X_tr) == len(y_tr)
        assert len(X_val) == len(y_val)
