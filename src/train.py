import os
import pandas as pd 
import numpy as np  
import joblib
import mlflow
import mlflow.lightgbm
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

def train():
    data_path = os.path.join("data", "creditcard.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Please download creditcard.csv from kaggle into csv to be able to use the data for training/")
    
    print("Dataset Loading......")
    df = pd.read_csv(data_path)

    # SPLITTING LABEL AND FEATURES
    x = df.drop(columns=["Class"])
    y = df["Class"]

    #SPLITTING TO MAINTAIN CLASS RATI0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    #CALCULATING SCALE WEIGHT FOR EXTREME CLASS IMBALANCE
    neg_count, pos_count = np.bincount(y_train)
    scale_weight = float(neg_count/pos_count)

    mlflow.set_experiment("Fraud-Detection-System")

    with mlflow.start_run(run_name="lgbm-baseline"):
        params ={
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.05,
            "scale_pos_weight": scale_weight,
            "random_state": 42,
            "n_jobs": -1
        }
        mlflow.log_params(params)

        print("Training LightGBM classifier...")
        model = LGBMClassifier(**params)
        model.fit(x_train, y_train)

        # Probabilities & Evaluation
        y_probs = model.predict_proba(x_test)[:, 1]
        threshold = 0.50
        y_preds = (y_probs >= threshold).astype(int)

        pr_auc = average_precision_score(y_test, y_probs)
        roc_auc = roc_auc_score(y_test, y_probs)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_preds, average="binary")

        metrics = {
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall)
        }
        mlflow.log_metrics(metrics)
        print(f"Metrics: {metrics}")

        # Save model artifact locally and in MLflow
        os.makedirs("models", exist_ok=True)
        artifact_path = os.path.join("models", "fraud_model.joblib")
        joblib.dump(model, artifact_path)
        mlflow.log_artifact(artifact_path)
        print(f"Model saved to {artifact_path}")

if __name__ == "__main__":
    train()