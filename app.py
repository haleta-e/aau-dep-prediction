from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# ---------------- HOME ----------------

@app.route('/')
def home():
    return render_template('senior_survey.html')

@app.route('/freshman')
def freshman():
    return render_template('freshman_survey.html')

# ---------------- SAVE SENIOR DATA ----------------

@app.route('/submit_senior', methods=['POST'])
def submit_senior():

    data = request.form.to_dict()

    # Automatically determine admission to preference
    if data["preferred_department"] == data["admitted_department"]:
        data["admitted_to_preference"] = 1
    else:
        data["admitted_to_preference"] = 0

    df = pd.DataFrame([data])

    if not os.path.exists("seniors.csv"):
        df.to_csv("seniors.csv", index=False)
    else:
        df.to_csv("seniors.csv", mode='a', header=False, index=False)

    return """
    Senior data saved successfully.
    <br><br>
    <a href="/freshman">Go to Prediction Page</a>
    """

# ---------------- ML PREDICTION ----------------

@app.route('/predict', methods=['POST'])
def predict():

    if not os.path.exists("seniors.csv"):
        return "No historical senior data available."

    seniors = pd.read_csv("seniors.csv")

    # Convert types
    seniors["gpa"] = seniors["gpa"].astype(float)
    seniors["seats"] = seniors["seats"].astype(int)
    seniors["total_applicants"] = seniors["total_applicants"].astype(int)
    seniors["admitted_to_preference"] = seniors["admitted_to_preference"].astype(int)

    # Create competition ratio
    seniors["competition_ratio"] = seniors["total_applicants"] / seniors["seats"]

    # ML Features
    X = seniors[["gpa", "competition_ratio"]]
    y = seniors["admitted_to_preference"]

    # Train Logistic Regression Model
    model = LogisticRegression()
    model.fit(X, y)

    # Print model parameters (educational purpose)
    print("Model Coefficients:", model.coef_)
    print("Model Intercept:", model.intercept_)

    # Get freshman input
    gpa = float(request.form["gpa"])
    seats = int(request.form["seats"])
    applicants = int(request.form["total_applicants"])
    dept = request.form["first_choice"]

    competition_ratio = applicants / seats

    # Predict Cut Point (minimum admitted GPA historically)
    dept_data = seniors[seniors["admitted_department"] == dept]

    if dept_data.empty:
        return "Not enough department data available."

    predicted_cutpoint = round(dept_data["gpa"].min(), 2)

    # ML Probability Prediction
    probability = model.predict_proba(
        [[gpa, competition_ratio]]
    )[0][1]

    percent = round(probability * 100, 2)

    # Competition Level
    if competition_ratio > 3:
        competition_level = "High"
    elif competition_ratio > 1.5:
        competition_level = "Medium"
    else:
        competition_level = "Low"

    return f"""
    <h2>Admission Prediction Result</h2>

    <p><strong>Predicted Cut Point:</strong> {predicted_cutpoint}</p>
    <p><strong>Competition Level:</strong> {competition_level}</p>
    <p><strong>Estimated Admission Probability:</strong> {percent}%</p>
    """

if __name__ == "__main__":
    app.run(debug=True)