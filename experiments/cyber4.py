import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv("UNSW_NB15/UNSW_NB15_training-set.csv")

# Preprocessing
label_encoder = LabelEncoder()
dataset['label'] = label_encoder.fit_transform(dataset['attack_cat'])

# Mapping numerical labels to attack categories and normal
label_mapping = dict(zip(dataset['label'], dataset['attack_cat']))
label_mapping[0] = "Normal"

# Feature selection
features = ['id','dur','spkts','dpkts','sbytes','dbytes','rate','sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit','swin','stcpb','dtcpb','dwin','tcprtt','synack',
            'ackdat','smean','dmean','trans_depth','response_body_len','ct_srv_src','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm',
                	'ct_srv_dst','is_sm_ips_ports']
# Define your features here

# Splitting the dataset
X = dataset[features]
y = dataset['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
linear_model = LogisticRegression(max_iter=1000)
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)
linear_accuracy = accuracy_score(y_test, linear_pred)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Train LightGBM
lgbm_model = LGBMClassifier()
lgbm_model.fit(X_train, y_train)
lgbm_pred = lgbm_model.predict(X_test)
lgbm_accuracy = accuracy_score(y_test, lgbm_pred)

# Train Decision Trees
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

# Model evaluation
print("Linear Regression Accuracy:", linear_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
print("LightGBM Accuracy:", lgbm_accuracy)
print("Decision Trees Accuracy:", dt_accuracy)

# Plotting the confusion matrix for Random Forest (you can do the same for other models)
plt.figure(figsize=(8, 6))
conf_matrix_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(conf_matrix_rf, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(ticks=range(len(label_mapping)), labels=label_mapping.values(), rotation=90)
plt.yticks(ticks=range(len(label_mapping)), labels=label_mapping.values(), rotation=0)
plt.show()
