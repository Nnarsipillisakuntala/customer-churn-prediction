# Data preprocessing for customer churn project
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
data = pd.read_csv("data/churn.csv")

# Remove customerID if present
if 'customerID' in data.columns:
    data = data.drop('customerID', axis=1)

# Convert TotalCharges to numeric
if 'TotalCharges' in data.columns:
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Fill missing values
data = data.fillna(0)

# Encode categorical columns
encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = encoder.fit_transform(data[column])

# Features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Preprocessing completed successfully")
