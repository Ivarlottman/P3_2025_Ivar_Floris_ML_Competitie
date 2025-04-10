"""
Model class van Ivar en Floris
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay
import joblib

class i_f_classy():
    """

    """
    def __init__(self, pipeline_path="model.joblib"):
        # Laad alle pipeline objecten:
        model_pipeline = joblib.load(pipeline_path) 

        # SVC model:
        self.model = model_pipeline["model"]

        # Imputers:
        self.ordinal_imputer = model_pipeline["ordinal_imputer"]
        self.continuous_imputer =  model_pipeline["continuous_imputer"]

        # Encoders:
        self.onehot_encoder =  model_pipeline["onehot_encoder"]
        self.continuous_scaler = model_pipeline["continuous_scaler"]

        # Features:
        self.features_ordinal = model_pipeline["features_ordinal"]
        self.features_continuous = model_pipeline["features_continuous"]
        self.all_features = self.features_ordinal + self.features_continuous


    def predict(self, filename="competition-train-noclass(1).csv"):
        """

        """

        mhs_df = pd.read_csv(filename)

        # Preprocessing: 
        mhs_df["sex"] = mhs_df["sex"].replace({"M": '0', "V": '1'})
        mhs_df[["slaapscore", "bmi"]] = mhs_df[["slaapscore", "bmi"]].round(1)

        X_not_imputed = mhs_df[self.all_features]


        X_ordinal = X_not_imputed[self.features_ordinal]

        X_continuous = X_not_imputed[self.features_continuous]

        
        X_ordinal_imputed = self.ordinal_imputer.transform(X_ordinal)

        X_ordinal_encoded = self.onehot_encoder.transform(X_ordinal_imputed)
        
        X_continuous_imputed = self.continuous_imputer.fit_transform(X_continuous)

        # Niet alle continue waarden hebben de zelfde eenheid, schaal ze:
        X_continuous_scaled = self.continuous_scaler.fit_transform(X_continuous_imputed)

        # Combineer features:
        X = np.hstack((X_ordinal_encoded, X_continuous_scaled))

        prediction = self.model.predict(X)
        #print(self.model)
        print(type(prediction))
        return prediction
    
if __name__ == '__main__':
    saved_model = "model_pipeline.joblib"
    test_data = "competition-train-noclass(1).csv"

    classifier = i_f_classy(pipeline_path=saved_model)
    prediction_results = classifier.predict(filename=test_data)
    print(prediction_results)