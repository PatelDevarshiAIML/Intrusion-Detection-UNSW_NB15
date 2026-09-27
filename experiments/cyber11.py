import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score, mean_absolute_error
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

# Reverse mapping for attack categories
attack_categories = dict(enumerate(label_encoder.classes_))

# Feature selection
features = ['id', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']

# Splitting the dataset
X_train = train_dataset[features]
y_train = train_dataset['label']
X_test = test_dataset[features]
y_test = test_dataset['label']

# Train models
models = {
    "Linear Regression": LogisticRegression(max_iter=10),
    "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42),
    "LightGBM": LGBMClassifier(),
    "Decision Trees": DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    # Save the trained model
    joblib.dump(model, f"{name}_model.pkl")

    # Evaluate the models
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    mae = mean_absolute_error(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    # Compute ROC curve and Precision-Recall curve
    n_classes = len(np.unique(y_test))

    # Plot ROC Curve
    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{attack_categories[i]} (AUC = {roc_auc_score(y_test == i, y_prob[:, i]):.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{name} ROC Curve (Multiclass)")
    plt.legend(loc="lower right")
    plt.show()

    # Plot Precision-Recall Curve
    plt.figure()
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test == i, y_prob[:, i])
        plt.plot(recall, precision, label=f"{attack_categories[i]}")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"{name} Precision-Recall Curve (Multiclass)")
    plt.legend(loc="lower left")
    plt.show()

    # Plot Mean Absolute Error (MAE)
    plt.figure()
    plt.bar('Mean Absolute Error', mae)
    plt.ylim([0.0, 1.0])
    plt.title(f"{name} Mean Absolute Error")
    plt.show()

    # Plot Precision
    plt.figure()
    plt.bar('Precision', precision)
    plt.ylim([0.0, 1.0])
    plt.title(f"{name} Precision")
    plt.show()

    # Plot Recall
    plt.figure()
    plt.bar('Recall', recall)
    plt.ylim([0.0, 1.0])
    plt.title(f"{name} Recall")
    plt.show()

    # Plot Accuracy
    plt.figure()
    plt.bar('Accuracy', accuracy)
    plt.ylim([0.0, 1.0])
    plt.title(f"{name} Accuracy")
    plt.show()

    # Plot F1-score
    plt.figure()
    plt.bar('F1-score', f1)
    plt.ylim([0.0, 1.0])
    plt.title(f"{name} F1-score")
    plt.show()

    # Display Confusion Matrix
    plt.figure()
    plt.imshow(conf_matrix, cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    plt.xticks(range(len(unique_labels)), [attack_categories[label] for label in unique_labels], fontsize=8, rotation=90)
    plt.yticks(range(len(unique_labels)), [attack_categories[label] for label in unique_labels], fontsize=8)
    for j in range(len(conf_matrix)):
        for k in range(len(conf_matrix)):
            plt.text(k, j, conf_matrix[j, k], ha="center", va="center", color="white" if conf_matrix[j, k] > conf_matrix.max() / 2 else "black", fontsize=8)
    plt.colorbar()
    plt.show()