# -*- coding: utf-8 -*-

"""
SPDX-FileCopyrightText: Group-3_RSE_TeamProject
SPDX-License-Identifier: CC0: Public Domain

This module performs model training, hyperparameter tuning, and evaluation
on the Disease Symptoms and Patient Profile dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import LabelEncoder


def encode_non_numerical(df):
    le = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = le.fit_transform(df[column])
    return df


def train_test_split_and_scale(df):
    X = df.drop(columns="Outcome Variable")
    y = df["Outcome Variable"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=9
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def get_model_config():
    return {
        "Logistic Regression": {
            "model": LogisticRegression(solver="saga", penalty="l2", max_iter=1000),
            "params": {"C": np.logspace(-3, 3, 7)},
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(),
            "params": {
                "criterion": ["gini", "entropy"],
                "max_depth": list(range(1, 21)),
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": [100, 200],
                "max_features": ["sqrt", "log2"],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
            },
        },
        "SVM": {
            "model": svm.SVC(probability=True),
            "params": {
                "kernel": ["linear", "rbf"],
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
            },
        },
        "NaiveBayes": {"model": GaussianNB(), "params": {}},
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
        },
        "Gradient Boost": {
            "model": GradientBoostingClassifier(),
            "params": {
                "learning_rate": [0.1, 0.2],
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
            },
        },
    }


def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    algorithms = get_model_config()
    best_model = {}
    best_model_details = []

    for model_name, config in algorithms.items():
        print(f"\n🔍 Tuning {model_name}...")
        rscv = RandomizedSearchCV(
            config["model"],
            config["params"],
            cv=5,
            n_iter=10,
            n_jobs=-1,
            verbose=0,
            random_state=42,
        )
        rscv.fit(X_train, y_train)

        best_model[model_name] = rscv
        best_model_details.append(
            {
                "Model Name": model_name,
                "Best Score": rscv.best_score_,
                "Best Parameters": rscv.best_params_,
            }
        )

    print("\n✅ Model tuning complete.")
    return best_model, pd.DataFrame(best_model_details)


def evaluate_metrics(best_model, X_train, y_train, X_test, y_test):
    train_metrics = []
    test_metrics = []

    for model_name, model in best_model.items():
        y_pred_train = model.predict(X_train)
        y_pred_train_probs = model.predict_proba(X_train)[:, 1]
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(
            y_train, y_pred_train
        ).ravel()

        y_pred_test = model.predict(X_test)
        y_pred_test_probs = model.predict_proba(X_test)[:, 1]
        tn_test, fp_test, fn_test, tp_test = confusion_matrix(
            y_test, y_pred_test
        ).ravel()

        train_metrics.append(
            {
                "Model Name": model_name,
                "MAE": mean_absolute_error(y_train, y_pred_train),
                "MSE": mean_squared_error(y_train, y_pred_train),
                "RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "AUC": roc_auc_score(y_train, y_pred_train_probs),
                "Sensitivity": tp_train / (tp_train + fn_train),
                "Specificity": tn_train / (tn_train + fp_train),
                "MCC": matthews_corrcoef(y_train, y_pred_train),
            }
        )

        test_metrics.append(
            {
                "Model Name": model_name,
                "MAE": mean_absolute_error(y_test, y_pred_test),
                "MSE": mean_squared_error(y_test, y_pred_test),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "AUC": roc_auc_score(y_test, y_pred_test_probs),
                "Sensitivity": tp_test / (tp_test + fn_test),
                "Specificity": tn_test / (tn_test + fp_test),
                "MCC": matthews_corrcoef(y_test, y_pred_test),
            }
        )

    return pd.DataFrame(train_metrics), pd.DataFrame(test_metrics)


def run_modeling_pipeline(df):
    print("\n📈 Running ML pipeline...")
    df_encoded = encode_non_numerical(df.copy())
    X_train, X_test, y_train, y_test = train_test_split_and_scale(df_encoded)
    best_models, best_model_details = train_and_evaluate_models(
        X_train, y_train, X_test, y_test
    )
    train_df, test_df = evaluate_metrics(best_models, X_train, y_train, X_test, y_test)

    print("\n-------- Best Hyperparameters --------")
    print(best_model_details)
    print("\n-------- Training Metrics --------")
    print(train_df)
    print("\n-------- Test Metrics --------")
    print(test_df)

    return best_models, train_df, test_df


if __name__ == "__main__":
    from eda import download_and_load_dataset, preprocess_dataset

    df = preprocess_dataset(download_and_load_dataset())
    run_modeling_pipeline(df)
