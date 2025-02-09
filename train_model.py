import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the preprocessed data.
df = pd.read_csv("preprocessed_transactions.csv")

# Separate the features (X) and the target label (y).
X = df.drop(columns=["fraud"])
y = df["fraud"]

# Split the data into training and testing sets (80% training, 20% testing).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model accuracy on the test set.
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}")

# Save the trained model.
joblib.dump(model, "iso20022_fraud_model.pkl")
