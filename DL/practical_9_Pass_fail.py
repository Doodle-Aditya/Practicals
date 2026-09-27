"""
Practical 9
TY BSC Data Science - Deep Learning
AIM: To develop a Student Pass/Fail Prediction System using Logistic 
Regression and integrate the trained machine-learning model with a Flask web 
application that predicts whether a student is likely to pass or fail based on their 
study hours.

Description: This project uses Logistic Regression to predict whether a student is 
likely to pass or fail based on study hours. The trained model is saved 
using pickle and integrated into a Flask web application for user-friendly predictions.
"""

import os
import pickle
from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression

# ==========================================
# 1. Model Training & Saving (train.py)
# ==========================================

# Training Data
X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

# Create model
model = LogisticRegression()

# Train model
model.fit(X, y)

# Save model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.pkl")

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully")

# ==========================================
# 2. Flask Application (app.py)
# ==========================================

# Configure template directory (supports templates/ subdirectory or root)
template_dir = os.path.join(base_dir, "templates")
if not os.path.exists(template_dir):
    template_dir = base_dir

app = Flask(__name__, template_folder=template_dir)

# Load trained model
with open(model_path, "rb") as file:
    loaded_model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def predict():
    result = ""

    if request.method == "POST":
        hours = float(request.form["hours"])

        prediction = loaded_model.predict([[hours]])

        if prediction[0] == 1:
            result = "Student is likely to PASS"
        else:
            result = "Student is likely to FAIL"

    return render_template(
        "index.html",
        result=result
    )


if __name__ == "__main__":
    # Verify predictions as shown in PDF output
    print("\n--- Output Verification ---")
    for hours in [4.0, 3.0]:
        pred = loaded_model.predict([[hours]])
        status = "Student is likely to PASS" if pred[0] == 1 else "Student is likely to FAIL"
        print(f"Enter Study Hours: {int(hours) if hours.is_integer() else hours} -> {status}")

    print("\nStarting Flask web application...")
    print("Open http://127.0.0.1:5000 in your browser to view the application.")
    app.run(debug=False)

# Put this in Index.html file 

# <html>
# <head>
# <title> Student Prediction </title>
# </head>
# <body>
# <h2> Student Pass/Fail Prediction </h2>
# <form method="POST">
# <label> Enter Study Hours: </label>
# <input type="number" name="hours" step="0.1" required>
# <button type="submit"> Predict </button>
# </form>
# <h3> {{result}} </h3>
# </body>
# </html>