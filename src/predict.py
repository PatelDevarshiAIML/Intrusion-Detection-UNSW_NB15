"""
Cybersecurity Attack Detection - Model Inference Script
Loads a trained ML model and performs prediction on network flow records.
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np

# 40 standard network features
FEATURE_COLUMNS = [
    'id', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
    'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
    'ct_srv_dst', 'is_sm_ips_ports'
]

# Attack categories mapping according to UNSW-NB15 label encoding
ATTACK_CLASSES = {
    0: 'Analysis',
    1: 'Backdoor',
    2: 'DoS',
    3: 'Exploits',
    4: 'Fuzzers',
    5: 'Generic',
    6: 'Normal',
    7: 'Reconnaissance',
    8: 'Shellcode',
    9: 'Worms'
}


def load_model(model_path: str):
    """Loads a serialized model artifact."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    print(f"Loading model from: {model_path}")
    return joblib.load(model_path)


def predict_sample(model, input_df: pd.DataFrame):
    """Generates predictions for a dataframe of flow samples."""
    # Ensure all required features are present
    for col in FEATURE_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = 0.0

    features = input_df[FEATURE_COLUMNS].fillna(0)
    preds = model.predict(features)

    # If model supports predict_proba, get confidence
    confidences = []
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)
        confidences = np.max(probs, axis=1)
    else:
        confidences = [1.0] * len(preds)

    results = []
    for pred_idx, conf in zip(preds, confidences):
        class_name = ATTACK_CLASSES.get(pred_idx, f"Class_{pred_idx}")
        results.append({
            "predicted_class_id": int(pred_idx),
            "predicted_category": class_name,
            "confidence": float(conf)
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference using trained UNSW-NB15 models")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser.add_argument("--model-path", type=str,
                        default=os.path.join(base_dir, "models", "DecisionTree_model.pkl"),
                        help="Path to trained model pickle")
    parser.add_argument("--input-csv", type=str, default=None,
                        help="Path to input CSV containing network flow records")

    args = parser.parse_args()

    # If no input CSV is provided, create a sample benign / attack test flow record
    if args.input_csv and os.path.exists(args.input_csv):
        input_data = pd.read_csv(args.input_csv)
    else:
        print("No input CSV provided. Using representative network traffic test sample.")
        sample_record = {
            'id': 1, 'dur': 0.001055, 'spkts': 2, 'dpkts': 2, 'sbytes': 132, 'dbytes': 164,
            'rate': 2843.60, 'sttl': 31, 'dttl': 29, 'sload': 500473.9, 'dload': 621800.9,
            'sloss': 0, 'dloss': 0, 'sinpkt': 0.017, 'dinpkt': 0.013, 'sjit': 0.0, 'djit': 0.0,
            'swin': 0, 'stcpb': 0, 'dtcpb': 0, 'dwin': 0, 'tcprtt': 0.0, 'synack': 0.0,
            'ackdat': 0.0, 'smean': 66, 'dmean': 82, 'trans_depth': 0, 'response_body_len': 0,
            'ct_srv_src': 3, 'ct_state_ttl': 0, 'ct_dst_ltm': 1, 'ct_src_dport_ltm': 1,
            'ct_dst_sport_ltm': 1, 'ct_dst_src_ltm': 2, 'is_ftp_login': 0, 'ct_ftp_cmd': 0,
            'ct_flw_http_mthd': 0, 'ct_src_ltm': 3, 'ct_srv_dst': 7, 'is_sm_ips_ports': 0
        }
        input_data = pd.DataFrame([sample_record])

    model = load_model(args.model_path)
    predictions = predict_sample(model, input_data)

    print("\nInference Results:")
    for idx, p in enumerate(predictions):
        print(f"Sample #{idx + 1}: Category -> {p['predicted_category']} (ID: {p['predicted_class_id']}, Confidence: {p['confidence']:.2%})")


if __name__ == "__main__":
    main()
