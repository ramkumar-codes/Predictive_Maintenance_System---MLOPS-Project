# src/train.py

import os
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=['Machine_ID', 'Maintenance_Needed'])
    y = df['Maintenance_Needed']
    return X, y


def plot_and_log_confusion_matrix(y_true, y_pred, labels, run_id):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    fname = f"confusion_matrix_{run_id}.png"
    plt.savefig(fname)
    plt.close()
    mlflow.log_artifact(fname)


def main():
    # Resolve paths relative to repo root
    base_path = Path(__file__).resolve().parent.parent
    data_path = base_path / 'data' / 'cnc_machine_data_multiclass.csv'
    print("Loading data from:", data_path)    # debug in CI logs
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    model_dir = base_path / 'model'
    model_dir.mkdir(exist_ok=True)

    # 1. Load & preprocess
    X, y = load_data(str(data_path))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 2. Configure MLflow (optional remote tracking)
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment('CNC_Maintenance_MLP_Multiclass')

    # 3. Start run and log everything
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Hyperparameters
        params = {
            'hidden_layer_sizes': (128, 64, 32),
            'max_iter': 500,
            'random_state': 42
        }
        mlflow.log_params(params)

        # Train model
        model = MLPClassifier(**params)
        model.fit(X_train, y_train)

        # Predict & evaluate
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_f1_macro': f1_score(y_test, y_test_pred, average='macro')
        }
        mlflow.log_metrics(metrics)

        # Log classification report
        report = classification_report(y_test, y_test_pred)
        with open('classification_report.txt', 'w') as f:
            f.write(report)
        mlflow.log_artifact('classification_report.txt')

        # Log confusion matrix
        plot_and_log_confusion_matrix(
            y_test, y_test_pred,
            labels=['Normal', 'Soon', 'Immediate'],
            run_id=run_id
        )

        # Log model with signature & example
        input_example = X_train[:5]
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='mlp_multiclass_model',
            signature=signature,
            input_example=input_example
        )

        # Save model+scaler locally
        with open(model_dir / 'mlp_cnc_model.pkl', 'wb') as f:
            pickle.dump((model, scaler), f)

    print(
        f"Run completed. View details with:\n"
        f"  mlflow ui --backend-store-uri {base_path}/mlruns\n"
        f"under run ID {run_id}"
    )


if __name__ == '__main__':
    main()
