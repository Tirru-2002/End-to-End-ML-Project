import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, predictPipeline
import os
import mysql.connector

application = Flask(__name__)
app = application

## Route for a home page
# @app.route('/')
# def index():
#     return render_template('index.html')g

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('writing_score')),
                writing_score=float(request.form.get('reading_score'))
            )
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            
            predict_pipeline = predictPipeline()
            
            results = predict_pipeline.predict(pred_df)
            print("Prediction Score")
            formatted_result = f"{results[0]:.2f}"

            # Save the data to a CSV file
            save_folder = 'StudentsData'
            os.makedirs(save_folder, exist_ok=True)
            file_path = os.path.join(save_folder, 'input_data.csv')
            pred_df['Math_prediction_score'] = formatted_result
            
            # Check if the file exists to determine if the header should be written
            if not os.path.isfile(file_path):
                pred_df.to_csv(file_path, index=False)
            else:
                pred_df.to_csv(file_path, mode='a', header=False, index=False)

                

            # Connect to MySQL database
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="DataForm"
            )
            cursor = conn.cursor()

            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS StudentsInputData (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    gender VARCHAR(255),
                    race_ethnicity VARCHAR(255),
                    parental_level_of_education VARCHAR(255),
                    lunch VARCHAR(255),
                    test_preparation_course VARCHAR(255),
                    reading_score FLOAT,
                    writing_score FLOAT,
                    math_prediction_score FLOAT
                )
            """)

            # Insert data into the table
            insert_query = """
                INSERT INTO StudentsInputData (
                    gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score, math_prediction_score
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                data.gender, data.race_ethnicity, data.parental_level_of_education, data.lunch, data.test_preparation_course, data.reading_score, data.writing_score, formatted_result
            ))

            # Commit the transaction
            conn.commit()

            # Close the connection
            cursor.close()
            conn.close()

            return render_template('home.html', results=formatted_result)
        
        except Exception as e:
            print(f"Error: {e}")
            return render_template('home.html', error="An error occurred during prediction. Please try again.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
