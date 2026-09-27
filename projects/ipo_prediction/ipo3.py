import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a pandas DataFrame
data = pd.read_csv('Dataset_Quarter.csv')

# Check for null values
print(data.isnull().sum())

# Drop rows with all null values
data = data.dropna(how='all')

# Group the data by 'CompanyName' and 'Subscription'
grouped = data.groupby(['CompanyName', 'Subscription']).size().reset_index(name='Count')

# Pivot the data to create a wide format
pivoted = grouped.pivot(index='CompanyName', columns='Subscription', values='Count').fillna(0)

# Sort the data by the sum of 'Count' columns
pivoted = pivoted.sort_values(by=[0, 1], ascending=False)

# Create a bar plot
fig, ax = plt.subplots(figsize=(12, 6))
pivoted.plot(kind='bar', ax=ax)

# Set the plot title and axis labels
ax.set_title('Distribution of Buy (1) and Sell (0) for Each Company')
ax.set_xlabel('Company Name')
ax.set_ylabel('Count')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=90)

# Add a legend
ax.legend(['Sell (0)', 'Buy (1)'], loc='upper right')

# Display the plot
plt.tight_layout()
plt.show()