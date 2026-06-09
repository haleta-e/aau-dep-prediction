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

    # auto label
    if data["preferred_department"] == data["admitted_department"]:
        data["admitted_to_preference"] = 1
    else:
        data["admitted_to_preference"] = 0

    df = pd.DataFrame([data])

    file_exists = os.path.exists("seniors.csv")
    df.to_csv("seniors.csv", mode='a', header=not file_exists, index=False)

    return """
    <h3>Senior data saved successfully</h3>
    <a href="/freshman">Go to Prediction Page</a>
    """

# ---------------- ML PREDICTION ----------------

@app.route('/predict', methods=['POST'])
def predict():

    if not os.path.exists("seniors.csv"):
        return "No data found"

    seniors = pd.read_csv("seniors.csv")

    # convert types
    seniors["gpa"] = seniors["gpa"].astype(float)
    seniors["entrance_exam_score"] = seniors["entrance_exam_score"].astype(float)
    seniors["seats"] = seniors["seats"].astype(int)
    seniors["total_applicants"] = seniors["total_applicants"].astype(int)
    seniors["admitted_to_preference"] = seniors["admitted_to_preference"].astype(int)

    # feature engineering
    seniors["competition_ratio"] = seniors["total_applicants"] / seniors["seats"]

    # features
    X = seniors[["gpa", "competition_ratio", "entrance_exam_score"]]
    y = seniors["admitted_to_preference"]

    model = LogisticRegression()
    model.fit(X, y)

    # user input
    gpa = float(request.form["gpa"])
    seats = int(request.form["seats"])
    applicants = int(request.form["total_applicants"])
    exam_score = float(request.form["entrance_exam_score"])

    competition_ratio = applicants / seats

    # prediction
    probability = model.predict_proba(
        [[gpa, competition_ratio, exam_score]]
    )[0][1]

    percent = round(probability * 100, 2)

    if competition_ratio > 3:
        competition_level = "High"
    elif competition_ratio > 1.5:
        competition_level = "Medium"
    else:
        competition_level = "Low"

    return f"""
    <h2>Admission Prediction Result</h2>
    <p><b>Admission Probability:</b> {percent}%</p>
    <p><b>Competition Level:</b> {competition_level}</p>
    """

if __name__ == "__main__":
    app.run(debug=True)
