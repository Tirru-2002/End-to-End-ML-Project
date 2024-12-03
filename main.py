from flask import Flask, request, render_template
import numpy as np
import pandas as pd

# Ensure the import paths match your project structure
from src.pipeline.predict_pipeline import CustomData, predictPipeline

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Collect form data
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),  # Fixed swapped scores
                writing_score=float(request.form.get('writing_score'))
            )
            
            # Convert data to DataFrame
            pred_df = data.get_data_as_data_frame()
            print("DataFrame for Prediction:")
            print(pred_df)

            # Predict using pipeline
            predict_pipeline = predictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            return render_template('home.html', results=results[0])
        
        except Exception as e:
            print(f"Error: {e}")
            return render_template('home.html', error="An error occurred during prediction. Please try again.")

if __name__ == "__main__":
    # For cloud deployment, use host='0.0.0.0' and a dynamic port
    app.run(host="0.0.0.0", port=8080)
