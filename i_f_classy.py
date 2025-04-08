"""
Model class van Ivar en Floris
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


class i_f_classy():
    """

    """
    def __init__(self):
        pass

    
    def predict(self, filename="competition-train(1).csv"):
        """

        """
        file = pd.read_csv(filename, skiprows=30)
        catagory_cols = ["opleidingsniveau", "sex"]
        nummeric_collumn = ["leeftijd", "lengte","gewicht", "bovendruk", "onderdruk", "tv-per-dag","beweging-per-week","slaapscore","maandinkomen","bmi","astigmatisme","hypertensie","MHS"]
        catagorical_frame = file.loc[:, catagory_cols]

        # encoding
        encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
        encoded_data = encoder.fit_transform(catagorical_frame)
        original_columns = file.loc[:,nummeric_collumn]
        # joind frame
        training_data = pd.concat([original_columns, encoded_data], axis=1)
        # x,y split
        x_col = training_data.columns.drop("MHS")
        x_data = training_data.loc[:,x_col].to_numpy()
        y_data = training_data["MHS"].to_numpy()

        # imputate x data
        feature_imputer = SimpleImputer(strategy='most_frequent')
        feature_imputer.fit(x_data)
        x_data_imputed = feature_imputer.transform(x_data)

        model = DecisionTreeClassifier().fit(x_data_imputed, y_data)
        prediction = model.predict(x_data_imputed)
        #print(prediction)
        print(type(prediction))
        return prediction
    
if __name__ == '__main__':
    model = i_f_classy()
    print(model.predict())