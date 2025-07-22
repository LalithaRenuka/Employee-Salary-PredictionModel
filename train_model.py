import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load data
df = pd.read_csv(r'C:\Users\lkoti\OneDrive\Documents\salary_predictor_india\Salary Dataset.csv')

# Select required columns only
df = df[['Company Name', 'Job Title', 'Experience', 'Location', 'Salary']]
df.dropna(inplace=True)

# Clean salary column: remove ₹, commas, and /yr
def clean_salary(s):
    try:
        s = s.replace('₹', '').replace(',', '').replace('/yr', '').strip()
        return float(s)
    except:
        return np.nan

df['Salary'] = df['Salary'].astype(str).apply(clean_salary)
df.dropna(subset=['Salary'], inplace=True)

# Create features and target
X = df[['Company Name', 'Job Title', 'Experience', 'Location']]
y = df['Salary']

# Preprocessing pipeline
categorical_cols = ['Company Name', 'Job Title', 'Location']
numerical_cols = ['Experience']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# Define model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X, y)

# Save model
joblib.dump(model, "salary_model.joblib")
print("✅ Model trained and saved as salary_model.joblib")
