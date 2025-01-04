<p align="center">
  <!-- <img src="preview/logo.png" alt="Logo" width="80" height="80"> -->

  <h3 align="center">Student Performance Prediction</h3>

  <p align="center">
    An end-to-end machine learning project predicting student math scores based on various factors.
    <br />
    <a href="https://github.com/Tirru-2002/End-to-End-ML-Project"><strong>Explore the Code »</strong></a>
    &nbsp; &nbsp;
    <a href="https://studentperformancemodel-1.et.r.appspot.com/"><strong>Cloud Model online »</strong></a>
  </p>
  
</p>



<!-- TABLE OF CONTENTS -->

## Table of Contents

1. [About the Project](#about-the-project)
    * [Built With](#built-with)
2. [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
3. [Usage](#usage)
4. [Cloud Deployment (Future)](#cloud-deployment-future)
5. [Roadmap](#roadmap)
6. [Contributing](#contributing)
7. [Contact](#contact)



<!-- ABOUT THE PROJECT -->

## About The Project

This project predicts student math scores using a machine learning model trained on a dataset containing various demographic and academic features.  The project follows a standard machine learning workflow:

* **Data Ingestion:** Data is loaded from a local CSV file or a MySQL database.
* **Data Transformation:**  Data preprocessing steps like handling missing values, encoding categorical features, and scaling numerical features are performed.
* **Model Training:**  Several regression models are evaluated (Random Forest, Decision Tree, Gradient Boosting, Linear Regression, XGBoost, CatBoost, AdaBoost, Ridge, SVR), and the best performing model is saved.
* **Model Deployment (Flask):** A Flask web application allows users to input student information and receive predicted math scores.  (Current deployment is local).
* **Error Handling and Logging:** Custom exceptions and logging are implemented for robust error management and tracking.


### Built With

* **Languages:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, CatBoost, Flask, MySQL Connector
* **Models:**  RandomForestRegressor, DecisionTreeRegressor, GradientBoostingRegressor, LinearRegression, XGBRegressor, CatBoostRegressor, AdaBoostRegressor, Ridge, SVR



<!-- GETTING STARTED -->

## Getting Started


### Prerequisites

* Python 3.7+
* Required libraries listed in `requirements.txt`

### Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/Tirru-2002/End-to-End-ML-Project.git
   ```
2. Create and activate a virtual environment (recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3. Install Dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Set up the database (if using MySQL - details in the code):
    XAMPP or MySql Workbench

## Usage

1. Redirect to components directory
   ```sh
   cd src/components/
   ```
2. Run the code file to save model
   ```sh
   python Ingestion_transform_train_fusion.py
   ```
3.  Run the Flask application: 
    ```sh 
    python main.py 
    ```
4. Access the web interface: http://127.0.0.1:5000/ (or the appropriate URL)

5. Input student details in the form to receive a math score prediction.


## Cloude Deployment (Future)

  This project is currently deployed locally using Flask. Future plans include deploying the model and application to Google Cloud Platform using services like:

  * Google App Engine or Google Kubernetes Engine: For hosting the Flask application.
  * Cloud SQL: For a managed MySQL database instance.
  * Vertex AI Endpoints or Cloud Functions: For potential serverless deployment of the prediction service. This would allow for scaling and management of the model in a production environment.

## Roadmap
  * Implement Cloud deployment (GCP).
  * Containerize the application using Docker.
  * Implement Continuous Integration/Continuous Deployment (CI/CD).
  * Add more data visualization and analysis.
  * Explore additional model optimization techniques.

## Contributing

  Contributions are welcome! Open an issue or submit a pull request


## Contact

  Linkedin - https://www.linkedin.com/in/tirumalchowdari


  Project Link: https://github.com/Tirru-2002/End-to-End-ML-Project

## References 

  Google cloud : <a href="https://cloud.google.com/appengine/docs/standard/python3/building-app/creating-gcp-project?_gl=1*si9s8h*_up*MQ..&gclid=ba20476e2733106a0ea32023c7e5d8a6&gclsrc=3p.ds">Create your project in gcloud</a>


