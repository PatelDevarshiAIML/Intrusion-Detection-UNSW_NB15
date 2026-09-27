import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
dataset = pd.read_csv("UNSW_NB15/UNSW_NB15_testing-set.csv")

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

# Train models
models = {
    "Linear Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=1000, random_state=42),
    "LightGBM": LGBMClassifier(num_leaves=31, max_depth=-1),
    "Decision Trees": DecisionTreeClassifier()
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    # Save the trained model
    joblib.dump(model, f"{name}_model.pkl")
    print(f"{name} trained and saved successfully")

# Print the accuracy of each model
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")
