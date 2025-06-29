# -*- coding: utf-8 -*-

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_model import (
    encode_non_numerical,
    train_test_split_and_scale,
    get_model_config,
    train_and_evaluate_models,
    evaluate_metrics
)
from eda import download_and_load_dataset, preprocess_dataset

@pytest.fixture(scope="module")
def sample_data():
    df = preprocess_dataset(download_and_load_dataset())
    return encode_non_numerical(df)

def test_data_encoding(sample_data):
    assert sample_data.isnull().sum().sum() == 0
    assert all(dtype != object for dtype in sample_data.dtypes)

def test_train_test_split(sample_data):
    X_train, X_test, y_train, y_test = train_test_split_and_scale(sample_data)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]

def test_model_config_structure():
    config = get_model_config()
    assert isinstance(config, dict)
    for name, meta in config.items():
        assert "model" in meta
        assert "params" in meta

def test_model_training_and_evaluation(sample_data):
    X_train, X_test, y_train, y_test = train_test_split_and_scale(sample_data)
    best_models, meta_df = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    assert not meta_df.empty
    assert all(model for model in best_models.values())

    train_metrics, test_metrics = evaluate_metrics(best_models, X_train, y_train, X_test, y_test)
    assert not train_metrics.empty
    assert not test_metrics.empty