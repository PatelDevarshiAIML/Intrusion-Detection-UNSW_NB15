import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the testing dataset
#testing_data = pd.read_csv("UNSW_NB15/UNSW_NB15_testing-set.csv")
testing_data = pd.read_csv("UNSW_NB15/UNSW_NB15_training-set.csv")
features = ['id','dur','spkts','dpkts','sbytes','dbytes','rate','sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack',
            'ackdat','smean','dmean','trans_depth','response_body_len','ct_srv_src','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm',
                	'ct_srv_dst','is_sm_ips_ports']
models = {
    "Linear Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LightGBM": LGBMClassifier(),
    "Decision Trees": DecisionTreeClassifier()
}
# Preprocessing for the testing data (same as training data preprocessing)
label_encoder = LabelEncoder()
testing_data['label'] = label_encoder.fit_transform(testing_data['attack_cat'])
X_test = testing_data[features]
y_test = testing_data['label']

# Evaluate the models on the testing data
for name, model in models.items():
    # Load the trained model
    trained_model = joblib.load(f"{name}_model.pkl")
    
    # Predict using the testing data
    y_pred = trained_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print accuracy
    print(f"{name} Accuracy: {accuracy}")
