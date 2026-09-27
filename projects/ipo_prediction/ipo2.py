import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load the data into a pandas DataFrame
data = pd.read_csv('Dataset_Quarter.csv')

# Check for null values
print(data.isnull().sum())

# Drop rows with all null values
data = data.dropna(how='all')

# Separate features and target variable
X = data.drop('CompanyName', axis=1)
y = data['CompanyName']

# Separate categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Impute missing values
categorical_imputer = SimpleImputer(strategy='most_frequent')
numerical_imputer = SimpleImputer(strategy='mean')

imputer = ColumnTransformer(transformers=[
    ('categorical', categorical_imputer, categorical_cols),
    ('numerical', numerical_imputer, numerical_cols)
])

X = imputer.fit_transform(X)

# One-hot encode categorical columns
encoder = OneHotEncoder(handle_unknown='ignore')
X = encoder.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred_proba = rf_model.predict_proba(X_test)

# Get unique company names
company_names = y_test.unique()

# Plot the ROC curve for each company name
plt.figure(figsize=(8, 6))
for i, company in enumerate(company_names):
    fpr, tpr, _ = roc_curve(y_test == company, y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{company} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Company Name')
plt.legend()
plt.show()