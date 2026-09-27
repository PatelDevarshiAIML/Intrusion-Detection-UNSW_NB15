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

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, mean_squared_error, classification_report

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Calculate precision and recall
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Generate classification report (including precision, recall, F1 score)
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Calculate Mean Squared Error (MSE) for the predicted probabilities
y_pred_proba = rf_model.predict_proba(X_test)


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


import numpy as np

# Plot accuracy, precision, recall
metrics = ['Accuracy', 'Precision', 'Recall']
values = [accuracy, precision, recall]

plt.figure(figsize=(8, 6))
plt.bar(metrics, values, color=['blue', 'green', 'orange'])
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Evaluation Metrics')
for i, value in enumerate(values):
    plt.text(i, value + 0.02, f'{value:.2f}', ha='center')
plt.show()


import seaborn as sns

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=company_names, yticklabels=company_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


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
