"""
Shared pytest fixtures.

Every fixture writes to pytest's tmp_path — nothing in data/ or models/ is touched.

Design notes
------------
- N_ENGINES=10, N_CYCLES=80: enough statistical power for select_features(threshold=0.2)
  to reliably find correlations with RUL in random data.
- feature_dir passes correlation_threshold=0.0 to build_features so feature
  selection always succeeds regardless of random seed.
- torch is NOT imported here — importing it at module level causes a segfault
  on macOS due to MPS/OpenMP conflicts with pytest's process. Test files that
  need torch import it inside their test functions or fixtures.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Sizing constants — large enough for feature selection to work reliably
# ---------------------------------------------------------------------------

N_ENGINES = 10
N_CYCLES = 80
N_FEATURES = 6


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _make_raw_train(
    n_engines: int = N_ENGINES, cycles_per_engine: int = N_CYCLES
) -> pd.DataFrame:
    """Synthetic FD001-like training data (space-separated, no header)."""
    rng = np.random.default_rng(42)
    rows = []
    for eid in range(1, n_engines + 1):
        for c in range(1, cycles_per_engine + 1):
            row = [eid, c]
            row += rng.uniform(-1, 1, 3).tolist()  # 3 settings
            row += rng.uniform(400, 700, 21).tolist()  # 21 sensors
            rows.append(row)
    cols = (
        ["engine_id", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )
    return pd.DataFrame(rows, columns=cols)


def _make_rul(n_engines: int = N_ENGINES) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"rul": rng.integers(10, 125, n_engines)})


def _make_feature_df(
    n_engines=N_ENGINES, n_cycles=N_CYCLES, n_features=N_FEATURES, include_rul=True
) -> pd.DataFrame:
    """
    Minimal feature-engineered DataFrame for in-memory tests.

    RUL decreases linearly per engine so at least some features
    can have non-zero correlation with it.
    """
    rng = np.random.default_rng(7)
    feat_cols = [f"f{i}" for i in range(n_features)]
    rows = []
    for eid in range(1, n_engines + 1):
        for c in range(1, n_cycles + 1):
            row = {"engine_id": eid, "cycle": c}
            # Mix signal + noise so correlation with RUL is detectable
            rul_val = max(0, n_cycles - c)
            row.update(
                {
                    feat_cols[0]: rul_val * 0.1 + rng.standard_normal(),  # correlated
                    **{f: rng.standard_normal() for f in feat_cols[1:]},  # noise
                }
            )
            if include_rul:
                row["rul"] = rul_val
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Raw file fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_dir(tmp_path):
    """Temp dir with synthetic FD001-format text files (no header, space-sep)."""
    train = _make_raw_train()
    test = _make_raw_train(cycles_per_engine=20)
    rul = _make_rul()

    train.to_csv(tmp_path / "train_FD001.txt", sep=" ", index=False, header=False)
    test.to_csv(tmp_path / "test_FD001.txt", sep=" ", index=False, header=False)
    rul.to_csv(tmp_path / "RUL_FD001.txt", sep=" ", index=False, header=False)
    return tmp_path


@pytest.fixture()
def raw_csv_dir(tmp_path, raw_dir):
    """Temp dir with load.py output: train.csv, test.csv, rul.csv."""
    from src.data.load import load_and_preprocess_data

    out = tmp_path / "raw"
    out.mkdir()
    load_and_preprocess_data(
        train_path=str(raw_dir / "train_FD001.txt"),
        test_path=str(raw_dir / "test_FD001.txt"),
        rul_path=str(raw_dir / "RUL_FD001.txt"),
        output_dir=out,
    )
    return out


@pytest.fixture()
def processed_dir(tmp_path, raw_csv_dir):
    """Temp dir with preprocess.py output: train_clean.csv, test_clean.csv, rul_clean.csv."""
    from src.data.preprocess import preprocess_data

    out = tmp_path / "processed"
    out.mkdir()
    preprocess_data(raw_dir=raw_csv_dir, processed_dir=out)
    return out


@pytest.fixture()
def feature_dir(tmp_path, processed_dir):
    """
    Temp dir with build_features output.

    Uses correlation_threshold=0.0 so feature selection always succeeds
    regardless of random seed in synthetic test data.
    """
    import shutil
    from src.features.build_feature import build_features

    out = tmp_path / "features"
    out.mkdir()
    for f in processed_dir.iterdir():
        shutil.copy(f, out / f.name)

    build_features(processed_dir=out, correlation_threshold=0.0)
    return out


@pytest.fixture()
def model_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# In-memory DataFrame fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_train_df():
    return _make_raw_train()


@pytest.fixture()
def raw_test_df():
    return _make_raw_train(cycles_per_engine=20)


@pytest.fixture()
def rul_df():
    return _make_rul()


@pytest.fixture()
def feature_train_df():
    return _make_feature_df(include_rul=True)


@pytest.fixture()
def feature_test_df():
    return _make_feature_df(include_rul=False)


@pytest.fixture()
def feat_cols():
    return [f"f{i}" for i in range(N_FEATURES)]
