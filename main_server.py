from fastapi import FastAPI
import pickle
app = FastAPI()
import pandas as pd

with open("ensemble_model.pkl", "rb") as f:
    model = pickle.load(f)


def preprocess_data(customer_data: dict):
    input_data = {
        "NumOfProducts": customer_data["NumOfProducts"],
        "IsActiveMember": int( customer_data["IsActiveMember"]),
        "Age": customer_data["Age"],
        "Geography_Germany": 1 if customer_data["Geography"] == "Germany" else 0,
        "Balance": customer_data["Balance"],
        "Geography_France": 1 if customer_data["Geography"] == "France" else 0,
        "Gender_Female": 1 if customer_data["Gender"] == "Female" else 0,
        "Geography_Spain": 1 if customer_data["Geography"] == "Spain" else 0,
        "CreditScore": customer_data["CreditScore"],
        "EstimatedSalary": customer_data["EstimatedSalary"],
        "HasCrCard": int(customer_data["HasCrCard"]),
        "Tenure": customer_data["Tenure"],
        "Gender_Male": 1 if customer_data["Gender"] == "Male" else 0,
        "AgeGroup_Elderly": 1 if customer_data["AgeGroup"] == "Elderly" else 0,
        "AgeGroup_MiddleAged": 1 if customer_data["AgeGroup"] == "MiddleAged" else 0,
        "AgeGroup_Senior": 1 if customer_data["AgeGroup"] == "Senior" else 0,
        "TenureAgeRatio": customer_data["Tenure"] / customer_data["Age"],
        "CLV": customer_data["CLV"]
    }

    customer_data_df = pd.DataFrame([input_data])
    print(customer_data_df)
    return customer_data_df

def get_prediction(customer_data: dict):
    preprocess_data_df = preprocess_data(customer_data)
    prediction = model.predict(preprocess_data_df)
    probability = model.predict_proba(preprocess_data_df)
    return prediction, probability

@app.post("/predict")
def predict(customer_data: dict):
    prediction, probability = get_prediction(customer_data)
    return {"prediction": prediction.tolist(), "probability": probability.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

customer_data = {
        "CreditScore": credit_score,
        "Geography": location,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_credit_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "CLV": clv,
        "AgeGroup": age_group,
        "TenureAgeRatio": tenure / age,
    }