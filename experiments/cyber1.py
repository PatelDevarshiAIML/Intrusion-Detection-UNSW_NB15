import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression  # Using logistic instead of linear regression for classification
from lightgbm import LGBMClassifier
# Load the dataset
dataset = pd.read_csv("UNSW_NB15/UNSW_NB15_training-set.csv")

# Preprocessing
label_encoder = LabelEncoder()
dataset['label'] = label_encoder.fit_transform(dataset['attack_cat'])

# Feature selection
features = ['id','dur','spkts','dpkts','sbytes','dbytes','rate','sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack',
            'ackdat','smean','dmean','trans_depth','response_body_len','ct_srv_src','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm',
                	'ct_srv_dst','is_sm_ips_ports']
# Define your features here

# Splitting the dataset
X = dataset[features]
y = dataset['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
models = {
    'RandomForest': RandomForestClassifier(),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'LightGBM': LGBMClassifier(random_state=42)
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = (model, accuracy, cm, report)
    print(f"{name} Model Accuracy: {accuracy:.4f}")
    
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes, rotation=0)
    plt.show()

# Get class names based on the label encoder
class_names = label_encoder.classes_

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plot_confusion_matrix(cm, class_names)

from sklearn.metrics import precision_recall_fscore_support as score

def plot_classification_report(y_true, y_pred):
    precision, recall, fscore, support = score(y_true, y_pred)
    report_df = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "support": support
    }, index=label_encoder.classes_)

    # Adding average / total row
    report_df.loc['avg / total'] = [
        np.average(precision, weights=support),
        np.average(recall, weights=support),
        np.average(fscore, weights=support),
        np.sum(support)
    ]

    plt.figure(figsize=(12, 6))
    sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap=plt.cm.Blues, fmt=".2f")
    plt.title("Classification Report")
    plt.show()

plot_classification_report(y_test, y_pred)


def plot_accuracy(accuracy):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=['Accuracy'], y=[accuracy])
    plt.ylim(0, 1)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.text(0, accuracy, f'{accuracy:.2f}', color='black', ha="center", va='bottom')
    plt.show()

plot_accuracy(accuracy)


def plot_classification_report(report, title='Classification Report'):
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(12, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap=plt.cm.Blues, fmt=".2f")  # skip the last row (accuracy)
    plt.title(title)
    plt.show()

# Iterate through results and plot metrics
for name, (model, accuracy, cm, report) in results.items():
    print(f"\n{name} Results:")
    plot_confusion_matrix(cm, class_names, title=f'{name} Confusion Matrix')
    plot_classification_report(report, title=f'{name} Classification Report')
    plot_accuracy(accuracy)

import pickle

# Save the trained models
for name, (model, _, _, _) in results.items():
    with open(f"{name}_model.pkl", "wb") as file:
        pickle.dump(model, file)
