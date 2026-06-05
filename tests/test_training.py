"""
tests/test_training.py — Training Tests

Tests cover train_baseline.py:
  - prepare_data produces correct shapes
  - each baseline model trains and returns predictions
  - evaluate() returns finite, non-negative RMSE and MAE
  - best model is saved to disk
  - full train_pipeline integration
"""

import numpy as np
import pandas as pd
import pytest
import joblib

from src.training.train_baseline import (
    evaluate,
    load_data,
    prepare_data,
    save_best_model,
    train_models,
    train_pipeline,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_engineered_dfs(n_engines=5, n_cycles=30, n_features=6):
    """Minimal feature-engineered DataFrames for training tests."""
    rng = np.random.default_rng(42)
    feat_cols = [f"f{i}" for i in range(n_features)]
    rows_train, rows_test = [], []

    for eid in range(1, n_engines + 1):
        for c in range(1, n_cycles + 1):
            row = {"engine_id": eid, "cycle": c}
            row.update({f: rng.standard_normal() for f in feat_cols})
            row["rul"] = max(0, n_cycles - c)
            rows_train.append(row)

        # Test: only last cycle per engine
        row_t = {"engine_id": eid, "cycle": n_cycles}
        row_t.update({f: rng.standard_normal() for f in feat_cols})
        rows_test.append(row_t)

    train_df = pd.DataFrame(rows_train)
    test_df  = pd.DataFrame(rows_test)
    rul_df   = pd.DataFrame({"rul": rng.integers(10, 100, n_engines)})
    return train_df, test_df, rul_df


# ===========================================================================
# prepare_data
# ===========================================================================


class TestPrepareData:

    def test_X_shape(self):
        train, test, rul = _make_engineered_dfs()
        X, y, X_test, y_test = prepare_data(train, test, rul)
        # All cycles, features only (no engine_id or rul)
        n_feat = len([c for c in train.columns if c not in ["engine_id", "rul"]])
        assert X.shape[1] == n_feat

    def test_X_test_one_row_per_engine(self):
        n_engines = 5
        train, test, rul = _make_engineered_dfs(n_engines=n_engines)
        _, _, X_test, y_test = prepare_data(train, test, rul)
        assert len(X_test) == n_engines
        assert len(y_test) == n_engines

    def test_no_engine_id_in_X(self):
        train, test, rul = _make_engineered_dfs()
        X, _, X_test, _ = prepare_data(train, test, rul)
        assert "engine_id" not in X.columns
        assert "engine_id" not in X_test.columns

    def test_no_rul_in_X(self):
        train, test, rul = _make_engineered_dfs()
        X, _, _, _ = prepare_data(train, test, rul)
        assert "rul" not in X.columns

    def test_y_non_negative(self):
        train, test, rul = _make_engineered_dfs()
        _, y, _, y_test = prepare_data(train, test, rul)
        assert (y >= 0).all()
        assert (y_test >= 0).all()


# ===========================================================================
# train_models
# ===========================================================================


class TestTrainModels:

    @pytest.fixture()
    def trained_models(self):
        train, test, rul = _make_engineered_dfs()
        X, y, _, _ = prepare_data(train, test, rul)
        return train_models(X, y), X, y

    def test_returns_all_four_models(self, trained_models):
        models, _, _ = trained_models
        assert set(models.keys()) == {"Linear", "Ridge", "Random Forest", "XGBoost"}

    def test_all_models_have_predict(self, trained_models):
        models, _, _ = trained_models
        for name, model in models.items():
            assert hasattr(model, "predict"), f"{name} has no predict method"

    @pytest.mark.parametrize("model_name", ["Linear", "Ridge", "Random Forest", "XGBoost"])
    def test_predictions_correct_shape(self, model_name):
        train, test, rul = _make_engineered_dfs()
        X, y, X_test, _ = prepare_data(train, test, rul)
        models = train_models(X, y)
        preds = models[model_name].predict(X_test)
        assert preds.shape == (len(X_test),)

    @pytest.mark.parametrize("model_name", ["Linear", "Ridge", "Random Forest", "XGBoost"])
    def test_predictions_finite(self, model_name):
        train, test, rul = _make_engineered_dfs()
        X, y, X_test, _ = prepare_data(train, test, rul)
        models = train_models(X, y)
        preds = models[model_name].predict(X_test)
        assert np.isfinite(preds).all(), f"{model_name} produced non-finite predictions"


# ===========================================================================
# evaluate
# ===========================================================================


class TestEvaluate:

    def test_returns_three_values(self):
        train, test, rul = _make_engineered_dfs()
        X, y, X_test, y_test = prepare_data(train, test, rul)
        models = train_models(X, y)
        result = evaluate(models["Linear"], X_test, y_test)
        assert len(result) == 3

    def test_rmse_non_negative(self):
        train, test, rul = _make_engineered_dfs()
        X, y, X_test, y_test = prepare_data(train, test, rul)
        models = train_models(X, y)
        rmse, mae, r2 = evaluate(models["XGBoost"], X_test, y_test)
        assert rmse >= 0
        assert mae  >= 0

    def test_perfect_predictions_zero_error(self):
        from sklearn.linear_model import LinearRegression
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.Series([2.0, 4.0, 6.0])
        model = LinearRegression().fit(X, y)
        rmse, mae, r2 = evaluate(model, X, y)
        assert rmse == pytest.approx(0.0, abs=1e-6)
        assert mae  == pytest.approx(0.0, abs=1e-6)
        assert r2   == pytest.approx(1.0, abs=1e-6)

    def test_r2_between_neg_inf_and_one(self):
        train, test, rul = _make_engineered_dfs()
        X, y, X_test, y_test = prepare_data(train, test, rul)
        models = train_models(X, y)
        _, _, r2 = evaluate(models["Ridge"], X_test, y_test)
        assert r2 <= 1.0


# ===========================================================================
# save_best_model
# ===========================================================================


class TestSaveBestModel:

    def test_file_created(self, model_dir):
        train, test, rul = _make_engineered_dfs()
        X, y, X_test, y_test = prepare_data(train, test, rul)
        models = train_models(X, y)
        results = pd.DataFrame([
            {"Model": "XGBoost", "RMSE_Test": 10.0},
            {"Model": "Linear",  "RMSE_Test": 15.0},
        ])
        name, path = save_best_model(models, results, output_dir=model_dir)
        assert path.exists(), f"Model file not created at {path}"

    def test_saved_model_is_loadable(self, model_dir):
        train, test, rul = _make_engineered_dfs()
        X, y, X_test, y_test = prepare_data(train, test, rul)
        models = train_models(X, y)
        results = pd.DataFrame([{"Model": "XGBoost", "RMSE_Test": 10.0}])
        _, path = save_best_model(models, results, output_dir=model_dir)
        loaded = joblib.load(path)
        assert hasattr(loaded, "predict")

    def test_best_model_is_lowest_rmse(self, model_dir):
        train, test, rul = _make_engineered_dfs()
        X, y, X_test, y_test = prepare_data(train, test, rul)
        models = train_models(X, y)
        results = pd.DataFrame([
            {"Model": "Ridge",  "RMSE_Test": 5.0},   # best
            {"Model": "Linear", "RMSE_Test": 20.0},
        ]).sort_values("RMSE_Test").reset_index(drop=True)
        name, _ = save_best_model(models, results, output_dir=model_dir)
        assert name == "Ridge"


# ===========================================================================
# train_pipeline integration
# ===========================================================================


class TestTrainPipelineIntegration:

    def test_returns_dataframe(self, feature_dir, model_dir):
        result = train_pipeline(processed_dir=feature_dir, model_dir=model_dir)
        assert isinstance(result, pd.DataFrame)

    def test_results_has_expected_columns(self, feature_dir, model_dir):
        result = train_pipeline(processed_dir=feature_dir, model_dir=model_dir)
        for col in ["Model", "RMSE_Val", "RMSE_Test", "MAE_Val", "MAE_Test"]:
            assert col in result.columns

    def test_results_csv_created(self, feature_dir, model_dir):
        train_pipeline(processed_dir=feature_dir, model_dir=model_dir)
        assert (model_dir / "baseline_results.csv").exists()

    def test_best_model_pkl_created(self, feature_dir, model_dir):
        train_pipeline(processed_dir=feature_dir, model_dir=model_dir)
        pkl_files = list(model_dir.glob("*.pkl"))
        assert len(pkl_files) >= 1, "No .pkl model file found after train_pipeline"

    def test_rmse_is_finite_and_positive(self, feature_dir, model_dir):
        result = train_pipeline(processed_dir=feature_dir, model_dir=model_dir)
        assert (result["RMSE_Test"] > 0).all()
        assert result["RMSE_Test"].apply(np.isfinite).all()