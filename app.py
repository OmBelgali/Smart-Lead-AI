from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained pipeline
pipeline = joblib.load('smartlead_pipeline.pkl')


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probabilities = None

    if request.method == "POST":
        # Get form data
        company = request.form.get("Company")
        industry = request.form.get("Industry")
        employees = request.form.get("Employees")
        funding = request.form.get("Funding")
        website_traffic = request.form.get("Website_Traffic")
        email_response = request.form.get("Email_Response_Rate")

        # Only proceed if all required fields are filled
        if all([company, industry, employees, funding, website_traffic, email_response]):
            # Create a DataFrame for prediction
            sample = pd.DataFrame([{
                "Company": company,
                "Industry": industry,
                "Employees": float(employees),
                "Funding": float(funding),
                "Website_Traffic": float(website_traffic),
                "Email_Response_Rate": float(email_response)
            }])

            # Predict lead quality
            prediction = pipeline.predict(sample)[0]

            # Get prediction probabilities if available
            # Get prediction probabilities if available
            if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
                probs_array = pipeline.predict_proba(sample)[0]
                class_labels = pipeline.named_steps['classifier'].classes_
                # Convert to dict and sort High → Medium → Low
                order = ["High", "Medium", "Low"]
                probabilities = {label: round(prob, 2) for label, prob in zip(class_labels, probs_array)}
                probabilities = {k: probabilities[k] for k in order if k in probabilities}

    return render_template("index.html", prediction=prediction, probabilities=probabilities)


if __name__ == "__main__":
    app.run(debug=True)
