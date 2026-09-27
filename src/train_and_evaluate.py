"""
Enhancing Cybersecurity with Machine Learning: Leveraging UNSW-NB15 Dataset
Main Training and Multi-Model Evaluation Pipeline
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution and server environments
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

# 40 core network traffic features selected from UNSW-NB15
FEATURE_COLUMNS = [
    'id', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
    'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
    'ct_srv_dst', 'is_sm_ips_ports'
]

TARGET_COLUMN = 'attack_cat'


def load_and_preprocess(train_path: str, test_path: str):
    """Loads training and test CSVs and applies feature selection and label encoding."""
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading testing data from: {test_path}")
    test_df = pd.read_csv(test_path)

    # Clean potential missing values in selected features
    available_features = [col for col in FEATURE_COLUMNS if col in train_df.columns]
    print(f"Using {len(available_features)} network features.")

    label_encoder = LabelEncoder()
    train_df['label'] = label_encoder.fit_transform(train_df[TARGET_COLUMN].astype(str))
    test_df['label'] = label_encoder.transform(test_df[TARGET_COLUMN].astype(str))

    X_train = train_df[available_features].fillna(0)
    y_train = train_df['label']
    X_test = test_df[available_features].fillna(0)
    y_test = test_df['label']

    return X_train, y_train, X_test, y_test, label_encoder, available_features


def get_models():
    """Initializes machine learning classification models."""
    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
    }


def plot_confusion_matrix(cm, classes, title, output_path):
    """Plots and saves a formatted confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_classification_report(report_dict, title, output_path):
    """Plots and saves the classification report as a heatmap."""
    report_df = pd.DataFrame(report_dict).transpose()
    plt.figure(figsize=(12, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Blues", fmt=".2f")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_metrics_summary(results, output_path):
    """Plots accuracy comparison bar chart for all models."""
    names = list(results.keys())
    accuracies = [results[n]['accuracy'] for n in names]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, accuracies, color='steelblue')
    plt.ylim(0, 1.05)
    plt.title('Model Accuracy Comparison', fontsize=14)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.xlabel('Model', fontsize=12)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.4f}",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def train_and_evaluate(train_path, test_path, output_dir, models_dir, selected_models=None):
    """Executes model training, evaluation, and saves visual results."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    X_train, y_train, X_test, y_test, label_encoder, features = load_and_preprocess(train_path, test_path)
    classes = label_encoder.classes_

    all_models = get_models()
    if selected_models:
        all_models = {k: v for k, v in all_models.items() if k in selected_models}

    evaluation_summary = {}

    print("\n" + "=" * 60)
    print("STARTING MODEL TRAINING & EVALUATION")
    print("=" * 60)

    for name, model in all_models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        rep = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)

        evaluation_summary[name] = {
            'accuracy': acc,
            'f1_macro': f1,
            'precision_macro': prec,
            'recall_macro': rec
        }

        print(f"{name} Results:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Macro F1 : {f1:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")

        # Save Visualizations
        cm_path = os.path.join(output_dir, f"ConfusionMatrix_{name}.png")
        plot_confusion_matrix(cm, classes, f"{name} Confusion Matrix", cm_path)

        rep_path = os.path.join(output_dir, f"ClassificationReport_{name}.png")
        plot_classification_report(rep, f"{name} Classification Report", rep_path)

        # Save Model Artifact
        model_file = os.path.join(models_dir, f"{name}_model.pkl")
        joblib.dump(model, model_file)
        print(f"  Saved model artifact to: {model_file}")

    # Summary plot
    summary_plot_path = os.path.join(output_dir, "Model_Accuracy_Comparison.png")
    plot_metrics_summary(evaluation_summary, summary_plot_path)

    # Save summary table
    summary_df = pd.DataFrame(evaluation_summary).transpose()
    summary_csv = os.path.join(output_dir, "metrics_summary.csv")
    summary_df.to_csv(summary_csv)
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print(summary_df)
    print(f"Summary metrics saved to: {summary_csv}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cybersecurity Attack Detection using UNSW-NB15")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser.add_argument("--train-path", type=str,
                        default=os.path.join(base_dir, "data", "UNSW_NB15_training-set.csv"),
                        help="Path to training CSV file")
    parser.add_argument("--test-path", type=str,
                        default=os.path.join(base_dir, "data", "UNSW_NB15_testing-set.csv"),
                        help="Path to testing CSV file")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(base_dir, "results", "figures"),
                        help="Directory to save evaluation figures")
    parser.add_argument("--models-dir", type=str,
                        default=os.path.join(base_dir, "models"),
                        help="Directory to save trained model pickles")
    parser.add_argument("--models", nargs="+", choices=["RandomForest", "DecisionTree", "KNN", "LightGBM", "LogisticRegression"],
                        help="Subset of models to train (default: all)")

    args = parser.parse_args()
    train_and_evaluate(args.train_path, args.test_path, args.output_dir, args.models_dir, args.models)
