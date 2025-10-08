import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load('smartlead_pipeline.pkl')

# Define new sample input (can be partial, missing columns will be filled)
sample = pd.DataFrame([{
    'Industry': 'Health',
    'Employees': 120,
    'Funding': 3_000_000,
    'Website_Traffic': 4500,
    'Email_Response_Rate': 0.12
    # 'Company' is missing, pipeline will handle it
}])

# Get expected columns from the pipeline's preprocessor
# Works if you used ColumnTransformer + OneHotEncoder
preprocessor = pipeline.named_steps['preprocessor']
categorical_features = preprocessor.transformers_[0][2]  # list of categorical columns
numeric_features = preprocessor.transformers_[1][2]      # list of numeric columns
expected_columns = categorical_features + numeric_features

# Fill missing columns with default values
for col in expected_columns:
    if col not in sample.columns:
        # Fill categorical columns with 'Unknown', numeric with 0
        if col in categorical_features:
            sample[col] = 'Unknown'
        else:
            sample[col] = 0

# Reorder columns to match training order (optional, safer)
sample = sample[expected_columns]

# Predict
prediction = pipeline.predict(sample)
print("Predicted Lead Quality:", prediction[0])

# Optional: get probabilities
if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
    proba = pipeline.predict_proba(sample)
    print("Prediction Probabilities:", proba[0])
