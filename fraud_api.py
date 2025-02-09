from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model and scaler.
model = joblib.load("iso20022_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

# Function to extract features from an ISO 20022 JSON transaction.
def extract_features(transaction):
    # Extract the raw feature values.
    transaction_amount = float(transaction["PmtInf"]["CdtTrfTxInf"]["Amt"]["InstdAmt"])
    high_risk_country = 1 if transaction["PmtInf"]["DbtrAcct"]["Id"]["IBAN"].startswith(("NG", "IR", "SY")) else 0
    sanctioned_entity = 1 if transaction["PmtInf"]["Dbtr"]["Id"] in ["BlacklistedID1", "BlacklistedID2"] else 0
    regulatory_code = 1 if transaction["PmtInf"]["CdtTrfTxInf"]["RgltryRptg"]["Cd"] == "AML" else 0
    amount_risk = 1 if transaction_amount > 5000 else 0

    # Assemble features into an array.
    features = np.array([
        transaction_amount,
        high_risk_country,
        sanctioned_entity,
        regulatory_code,
        amount_risk
    ]).reshape(1, -1)
    
    # Scale the features using the previously saved scaler.
    features_scaled = scaler.transform(features)
    return features_scaled

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = extract_features(data)
        
        # Get probability estimates from the model.
        probabilities = model.predict_proba(features)
        # Calculate the risk score as the fraud probability (percentage).
        risk_score = probabilities[0][1] * 100
        
        # Get a binary prediction (0 or 1).
        prediction = model.predict(features)[0]
        
        response = {
            "fraud_detected": bool(prediction),
            "risk_score": f"{risk_score:.1f}%",
            "message": "⚠️ Suspicious transaction detected!" if prediction else "✅ Transaction is safe"
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
