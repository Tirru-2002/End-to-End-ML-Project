import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import mysql.connector
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import Ridge
from sklearn.svm import SVR


from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.utils import save_object,evaluate_models

# Assume CustomException, logging, save_object, and evaluate_models are imported from the src module.

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            try:
                df = pd.read_csv('notebook/data/stud.csv')
                logging.info('Read the dataset as dataframe from local CSV')

            except FileNotFoundError:  # Handle the case where the CSV isn't found
                logging.info("Local CSV not found, attempting to connect to MySQL.")

                conn = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="",
                    database="StudentsData"
                ) # Db not Created
                df = pd.read_sql_query("select * from `StudentsData`", con=conn)
                logging.info('Read the dataset as dataframe from MySQL')
                conn.close()  # Close the connection after reading data

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e,sys)



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"
            ]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False)),
            ])
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipelines", cat_pipeline, categorical_columns),
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
                
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                    f"Applying preprocessing object on training dataframe and testing dataframe."
                )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e, sys)




@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:


            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Ridge": Ridge(),
                "SVR": SVR(),
                
            }

            params = {
                "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
                "Decision Tree": {
                        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression":{},
                "XGBRegressor": {'learning_rate': [.1, .01, .05, .001], 'n_estimators': [8, 16, 32, 64,128,256]},
                "CatBoosting Regressor": {'depth': [6, 8,10], 'learning_rate': [0.01, 0.05,0.1], 'iterations': [30,50, 100]},
                "AdaBoost Regressor": {'learning_rate': [.1, .01,0.5,.001], 'n_estimators': [8, 16, 32,64,128,256]},
                "Ridge": {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'fit_intercept': [True, False],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
                    'tol': [1e-4, 1e-3, 1e-2],
                    'max_iter': [None, 100, 500, 1000],
                },
                "SVR": {
                    "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "C": [0.1, 1, 10, 100],
                    "epsilon": [0.1, 0.2, 0.5, 1],
                    "gamma": ["scale", "auto"]
                },
                
                
                
            
                
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No satisfactory model found.")
            print("Model Report :", model_report," ")
            print("Best Model :", best_model_name," ")
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            predicted = best_model.predict(X_test)
            return r2_score(y_test, predicted)
        
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(f"Model Score : {model_trainer.initiate_model_trainer(train_arr, test_arr):.4f}")
