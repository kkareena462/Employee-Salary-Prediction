import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_model():
    df = pd.read_csv("dataset/employee_data.csv")

    # Columns to encode
    cat_cols = ['Education_Level', 'Job_Title', 'Location', 'Company_Size', 'Certifications']
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le  # Save encoder for this column

    # Save all encoders together
    joblib.dump(encoders, "label_encoders.pkl")

    X = df.drop("Salary", axis=1)
    y = df["Salary"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "salary_model.pkl")
    print("✅ Model and encoders trained and saved.")
