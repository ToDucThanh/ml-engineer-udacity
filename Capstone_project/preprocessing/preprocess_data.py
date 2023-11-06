import pandas as pd

from sklearn.preprocessing import LabelEncoder
from typing import List

def encode_categorical_features(df: pd.DataFrame, categorical_features: List[str]):
    """Label Encode categorical features

    Args:
        df (pd.DataFrame): The dataset.
        categorical_features (List[str]): List of categorical features.
    """
    
    df1 = df.copy()
    for feature in categorical_features:
        le = LabelEncoder()
        df1[feature] = le.fit_transform(df1[feature])
    
    return df1
    