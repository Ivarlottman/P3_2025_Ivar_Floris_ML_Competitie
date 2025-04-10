"""
Model class van Ivar en Floris
MHA predictor
Datum: 10-04-2025
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer
from sklearn.svm import SVC
import joblib


class classified_information():


    def __init__(self, pipeline_path="model_pipeline.joblib"):

        # Laad alle pipeline objecten:
        model_pipeline = joblib.load(pipeline_path) 

        # SVC model: SVC(C=0.03, class_weight='balanced', kernel='linear', tol=0.1)
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
            Voorspel de uitkomst van de "MHS" kolom van de gegeven test dataset.
        """

        mhs_df = pd.read_csv(filename)

        # Preprocessing: 
        mhs_df["sex"] = mhs_df["sex"].replace({"M": '0', "V": '1'})
        mhs_df[["slaapscore", "bmi"]] = mhs_df[["slaapscore", "bmi"]].round(1)

        X_not_imputed = mhs_df[self.all_features]

        X_ordinal = X_not_imputed[self.features_ordinal]
        X_continuous = X_not_imputed[self.features_continuous]


        # Imputeren van NA's:
        X_ordinal_imputed = self.ordinal_imputer.transform(X_ordinal)
        X_continuous_imputed = self.continuous_imputer.fit_transform(X_continuous)

        # Onehot encode de catagrorische ordinale features:
        X_ordinal_encoded = self.onehot_encoder.transform(X_ordinal_imputed)

        # Niet alle continue waarden hebben de zelfde eenheid, schaal ze:
        X_continuous_scaled = self.continuous_scaler.fit_transform(X_continuous_imputed)

        # Combineer features:
        X = np.concatenate((X_ordinal_encoded, X_continuous_scaled), axis=1)

        # Voorspel en zet om in boolean array:
        prediction = self.model.predict(X)
        prediction = prediction.astype(dtype=bool)


        #print(self.model)
        #print(type(prediction))
        return prediction
    
if __name__ == '__main__':
    
    test_data = "competition-train-noclass(1).csv"

    prediction_results = classified_information().predict(filename=test_data)

    print(prediction_results)