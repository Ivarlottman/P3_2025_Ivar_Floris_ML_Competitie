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
    def __init__(self, model_path="model.joblib"):
        self.model = joblib.load(model_path)         
    


    def predict(self, filename="competition-train-noclass(1).csv"):
        """

        """

        mhs_df = pd.read_csv(filename)

        # Preprocessing: 
        mhs_df["sex"] = mhs_df["sex"].replace({"M": '0', "V": '1'})
        mhs_df[["slaapscore", "bmi"]] = mhs_df[["slaapscore", "bmi"]].round(1)

        # Features selecteren:
        features_ordinal = ["beweging-per-week", "tv-per-dag","sex", "astigmatisme", "hypertensie"]
        features_continuous = ["bmi", "leeftijd", "lengte", "bovendruk", "onderdruk", "slaapscore", "gewicht", "maandinkomen"]

        to_predict = ["MHS"]

        all_features = features_ordinal + features_continuous

        y = mhs_df[to_predict]
        X_not_imputed = mhs_df[all_features]


        X_ordinal = X_not_imputed[features_ordinal]
        X_continuous = X_not_imputed[features_continuous]

        ordinal_feature_imputer = SimpleImputer(strategy='most_frequent')
        X_ordinal_imputed = ordinal_feature_imputer.fit_transform(X_ordinal)

        onehot_encoder = OrdinalEncoder(
            categories=[[1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7], [0,1], [0,1], [0,1]],
            handle_unknown='use_encoded_value', 
            unknown_value=-1
        )

        X_ordinal_encoded = onehot_encoder.fit_transform(X_ordinal_imputed)

        continuous_imputer = KNNImputer(n_neighbors=2, weights="uniform")
        X_continuous_imputed = continuous_imputer.fit_transform(X_continuous)

        # Niet alle continue waarden hebben de zelfde eenheid, schaal ze:
        continuous_scaler = StandardScaler()
        X_continuous_scaled = continuous_scaler.fit_transform(X_continuous_imputed)

        # Combineer features:
        X = np.hstack((X_ordinal_encoded, X_continuous_scaled))

        prediction = self.model.predict(X)

        #print(type(prediction))
        return prediction
    
if __name__ == '__main__':
    saved_model = "super_model.joblib"
    test_data = "competition-train-noclass(1).csv"

    classifier = i_f_classy(model_path=saved_model)
    prediction_results = classifier.predict(filename=test_data)
    print(prediction_results)