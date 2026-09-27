import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Load the dataset
train_dataset = pd.read_csv("UNSW_NB15/UNSW_NB15_training-set.csv")
test_dataset = pd.read_csv("UNSW_NB15/UNSW_NB15_testing-set.csv")

# Preprocessing
label_encoder = LabelEncoder()
train_dataset['label'] = label_encoder.fit_transform(train_dataset['attack_cat'])
test_dataset['label'] = label_encoder.transform(test_dataset['attack_cat'])

# Feature selection
features = ['id', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']

# Splitting the dataset
X_train = train_dataset[features]
y_train = train_dataset['label']
X_test = test_dataset[features]
y_test = test_dataset['label']

# Train models
models = {
    "Linear Regression": LogisticRegression(max_iter=100),
    "Random Forest": RandomForestClassifier(n_estimators=1000, random_state=42),
    "LightGBM": LGBMClassifier(),
    "Decision Trees": DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    # Save the trained model
    joblib.dump(model, f"{name}_model.pkl")

# Evaluate the models
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.ravel()
for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

    # Display accuracy and F1 score
    axs[i].text(0.05, 0.85, f"Accuracy: {accuracy:.2f}", transform=axs[i].transAxes, fontsize=12)
    axs[i].text(0.05, 0.75, f"F1 Score: {f1:.2f}", transform=axs[i].transAxes, fontsize=12)
    axs[i].text(0.05, 0.65, f"ROC AUC: {roc_auc:.2f}", transform=axs[i].transAxes, fontsize=12)

    # Display confusion matrix
    axs[i].imshow(conf_matrix, cmap='Blues')
    axs[i].set_title(f"{name} Confusion Matrix")
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    axs[i].set_xticks(range(len(unique_labels)))
    axs[i].set_yticks(range(len(unique_labels)))
    axs[i].set_xticklabels(unique_labels, fontsize=8)
    axs[i].set_yticklabels(unique_labels, fontsize=8)
    for j in range(len(conf_matrix)):
        for k in range(len(conf_matrix)):
            axs[i].text(k, j, conf_matrix[j, k], ha="center", va="center", color="white" if conf_matrix[j, k] > conf_matrix.max() / 2 else "black", fontsize=8)

plt.tight_layout()
plt.show()