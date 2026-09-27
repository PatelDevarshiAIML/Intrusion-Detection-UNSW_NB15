import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv("UNSW_NB15/train_test_network.csv")

# Preprocessing
label_encoder = LabelEncoder()
dataset['label'] = label_encoder.fit_transform(dataset['type'])

# Feature selection
features = ['src_ip','src_port','dst_ip','dst_port','proto','service','duration','src_bytes','dst_bytes','conn_state','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes','dns_query','dns_qclass','dns_qtype','dns_rcode','http_request_body_len','http_response_body_len','http_status_code']
# Define your features here

# Splitting the dataset
X = dataset[features]
y = dataset['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Linear Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LightGBM": LGBMClassifier(),
    "Decision Trees": DecisionTreeClassifier()
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(list(accuracies.keys()), list(accuracies.values()), color='skyblue')
plt.xlabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.xlim(0, 1)
plt.gca().invert_yaxis()  # Invert y-axis to display models from top to bottom
plt.show()
