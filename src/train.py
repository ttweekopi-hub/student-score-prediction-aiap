import pandas as pd # Data manipulation
from sklearn.model_selection import train_test_split # To split data
from sklearn.ensemble import RandomForestRegressor # The chosen ML algorithm
from sklearn.preprocessing import OneHotEncoder # To handle categorical text
from sklearn.compose import ColumnTransformer # To apply different fixes to columns
from sklearn.pipeline import Pipeline # To chain steps together
import joblib # To save the trained model file
import os

# Load the cleaned data from the previous step
df = pd.read_csv('data/cleaned_score.csv')

# Define Features (X) and Target (y)
X = df.drop(columns=['final_test']) # Everything except the score
y = df['final_test'] # The score we want to predict

# Identify categorical columns for OneHotEncoding
cat_cols = ['tuition', 'learning_style', 'direct_admission', 'CCA', 'gender']

# Create a preprocessor that converts text to numbers
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), cat_cols) # Encode categories
    ], remainder='passthrough') # Keep numerical columns as they are

# Create a full pipeline: Preprocess -> Train
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), # Step 1: Encode categories
    ('regressor', RandomForestRegressor(n_estimators=100)) # Step 2: Train Forest
])

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train) # Fit the pipeline to training data

# ... your existing training code ...

# Create the models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')
    print("Created missing 'models' directory.")

# Now save the model
joblib.dump(model_pipeline, 'models/student_model.pkl')
print("Model Training Complete: Saved to models/student_model.pkl")

# Save the model to the models/ folder
joblib.dump(model_pipeline, 'models/student_model.pkl')
print("Model Training Complete: Saved to models/student_model.pkl")