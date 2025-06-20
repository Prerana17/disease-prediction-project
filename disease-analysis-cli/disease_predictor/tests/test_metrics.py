from disease_predictor.metrics import evaluate_model
from disease_predictor.trainer import split_and_scale, train_best_model
from disease_predictor.model_registry import get_models
from disease_predictor.data_loader import load_and_preprocess_data
import pytest

@pytest.mark.unit
@pytest.mark.functional
def test_model_metrics():
    X, y = load_and_preprocess_data("data/disease_dataset.csv")
    X_train, X_test, y_train, y_test = split_and_scale(X, y, test_size=0.25, seed=42)
    model = train_best_model(X_train, y_train, get_models()['SVM'])
    report, matrix = evaluate_model(model, X_test, y_test)
    assert isinstance(report, str)
    assert matrix.shape == (2, 2)
