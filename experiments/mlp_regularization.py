import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import train_test_split
from src.preprocessing import HeartDiseasePreprocessor
from src.metrics import calculate_metrics
from src.models.neural_network import MLPClassifier

def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    train_metrics = calculate_metrics(
        y_train,
        y_train_pred,
        y_train_proba
    )

    test_metrics = calculate_metrics(
        y_test,
        y_test_pred,
        y_test_proba
    )

    return train_metrics, test_metrics


def main():
    data_path = PROJECT_ROOT / "data" / "heart_cleveland_upload.csv"
    if not data_path.exists():
        data_path = Path("../data/heart_cleveland_upload.csv")
    df = pd.read_csv(data_path)

    X = df.drop("condition", axis=1)
    y = df["condition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = HeartDiseasePreprocessor()

    X_train_proc = preprocessor.fit_transform(
        X_train
    )

    X_test_proc = preprocessor.transform(
        X_test
    )

    configs = [
        {
            "name": "Baseline",
            "weight_decay": 0.0,
            "dropout_rate": 0.0
        },
        {
            "name": "Weight Decay",
            "weight_decay": 0.01,
            "dropout_rate": 0.0
        },
        {
            "name": "Dropout",
            "weight_decay": 0.0,
            "dropout_rate": 0.2
        },
        {
            "name": "Weight Decay + Dropout",
            "weight_decay": 0.01,
            "dropout_rate": 0.2
        }
    ]

    results = []
    histories = {}

    for config in configs:
        model = MLPClassifier(
            hidden_layers=(32, 16),
            learning_rate=0.01,
            epochs=300,
            batch_size=32,
            weight_decay=config["weight_decay"],
            dropout_rate=config["dropout_rate"],
            random_state=42
        )

        model.fit(
            X_train_proc,
            y_train
        )

        train_metrics, test_metrics = evaluate_model(
            model,
            X_train_proc,
            y_train,
            X_test_proc,
            y_test
        )

        histories[config["name"]] = (
            model.loss_history_
        )

        results.append({
            "Model": config["name"],
            "Weight_Decay": config["weight_decay"],
            "Dropout": config["dropout_rate"],
            "Train_Accuracy": train_metrics["Accuracy"],
            "Test_Accuracy": test_metrics["Accuracy"],
            "Train_F1": train_metrics["F1_Score"],
            "Test_F1": test_metrics["F1_Score"],
            "Train_ROC_AUC": train_metrics["ROC_AUC"],
            "Test_ROC_AUC": test_metrics["ROC_AUC"]
        })

    results_df = pd.DataFrame(results)

    print("\nMLP Regularization Comparison")
    print("=" * 80)
    print(results_df.to_string(index=False))

    plt.figure(figsize=(9, 5))

    for name, history in histories.items():
        plt.plot(
            history,
            label=name
        )

    plt.title(
        "MLP: Training Loss with Regularization"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()