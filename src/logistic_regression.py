# Logistic Regression model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("data/churn.csv")

if 'customerID' in data.columns:
    data = data.drop('customerID', axis=1)

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.fillna(0)

# Encode text columns
encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = encoder.fit_transform(data[col])

X = data.drop("Churn", axis=1)
y = data["Churn"]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train,y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test,y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
