from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)


model = joblib.load("rf_waste_model.pkl")

charities_df = pd.read_csv("charities.csv")


def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k,v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    else:
        return obj


def generate_waste_advice(food_type, quantity):
    advice = []
    if food_type.lower() in ["meat", "fish", "chicken"]:
        advice.append("Store proteins in freezer to avoid spoilage")
        advice.append("Cut large portions into small meals for easier consumption")
    elif food_type.lower() in ["vegetables", "fruits"]:
        advice.append("Store vegetables/fruits in fridge away from humidity")
        advice.append("Use edible parts for cooking or juices")
    elif food_type.lower() in ["bread", "pastry"]:
        advice.append("Freeze bread for later use")
        advice.append("Cut old bread into crumbs or baked goods")
    elif food_type.lower() in ["dairy", "cheese", "milk"]:
        advice.append("Store dairy products at proper temperature")
        advice.append("Use small portions first to avoid expiration")
    else:
        advice.append("Store food properly and control quantity")
    
    if quantity > 100:
        advice.append("Split large quantity over several days to reduce waste")
    elif quantity < 20:
        advice.append("Prepare small quantity first to avoid surplus")

    return " | ".join(advice)

@app.route("/")
def home():
    return render_template_string()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    will_waste = model.predict(df)[0]
    prob_waste = model.predict_proba(df)[:,1][0]

    match = "No matching charity because no waste expected"
    advice = "No advice needed"

    if will_waste == 1:
        total_quantity = df.loc[0, "per_guest_quantity"] * df.loc[0, "number_of_guests"]
        advice = generate_waste_advice(df.loc[0, "type_of_food"], total_quantity)

        candidate_charities = charities_df[charities_df["food_type"] == df.loc[0, "type_of_food"]].copy()
        if not candidate_charities.empty:
            candidate_charities["score"] = candidate_charities["location"].apply(
                lambda x: 1 if x == df.loc[0, "geographical_location"] else 0
            )
            best_charity = candidate_charities.sort_values("score", ascending=False).iloc[0]
            distribute_qty = min(total_quantity, best_charity.get("needed_quantity", 0))
            match = {
                "charity_name": best_charity["NGO_name"],
                "charity_location": best_charity["location"],
                "quantity_to_send": distribute_qty,
                "contact": best_charity["contact"]
            }
            match = convert_to_native(match)

    return jsonify({
        "will_waste": int(will_waste),
        "probability": float(prob_waste),
        "matching": match,
        "advice": advice
    })

if __name__ == "__main__":
    app.run(debug=True)
