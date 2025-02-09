import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load ISO 20022 JSON transactions
with open("transactions.json", "r") as file:
    transactions = json.load(file)

def extract_features_and_label(transaction):
    # Extract transaction amount as a float.
    transaction_amount = float(transaction["PmtInf"]["CdtTrfTxInf"]["Amt"]["InstdAmt"])
    
    # Feature: high_risk_country (1 if IBAN starts with a high-risk prefix)
    high_risk_country = 1 if transaction["PmtInf"]["DbtrAcct"]["Id"]["IBAN"].startswith(("NG", "IR", "SY")) else 0
    
    # Feature: sanctioned_entity (1 if debtor ID is blacklisted)
    sanctioned_entity = 1 if transaction["PmtInf"]["Dbtr"]["Id"] in ["BlacklistedID1", "BlacklistedID2"] else 0
    
    # Feature: regulatory_code (1 if regulatory reporting code equals "AML")
    regulatory_code = 1 if transaction["PmtInf"]["CdtTrfTxInf"]["RgltryRptg"]["Cd"] == "AML" else 0
    
    # New Feature: amount_risk (1 if transaction amount exceeds $5000)
    amount_risk = 1 if transaction_amount > 5000 else 0
    
    features = {
        "transaction_amount": transaction_amount,
        "high_risk_country": high_risk_country,
        "sanctioned_entity": sanctioned_entity,
        "regulatory_code": regulatory_code,
        "amount_risk": amount_risk
    }
    # The fraud label is included in the JSON (defaulting to 0 if missing).
    label = transaction.get("fraud", 0)
    return features, label

# Extract features and labels from each transaction.
features_list = []
labels_list = []
for tx in transactions:
    features, label = extract_features_and_label(tx)
    features_list.append(features)
    labels_list.append(label)

# Create a DataFrame from the extracted features.
df = pd.DataFrame(features_list)
df["fraud"] = labels_list

# Scale the features (improves model performance).
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop(columns=["fraud"])), columns=df.columns[:-1])
df_scaled["fraud"] = df["fraud"]

# Save the preprocessed data and the scaler.
df_scaled.to_csv("preprocessed_transactions.csv", index=False)
joblib.dump(scaler, "scaler.pkl")

print("✅ Data preprocessing completed! Saved as 'preprocessed_transactions.csv' and scaler as 'scaler.pkl'.")
