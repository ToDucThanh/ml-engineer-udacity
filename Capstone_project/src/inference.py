import os
import joblib
import pandas as pd
import json

from sklearn.preprocessing import LabelEncoder

JSON_CONTENT_TYPE = 'application/json'

def model_fn(model_dir):
    with open(os.path.join(model_dir, "et_clf.joblib"), "rb") as f:
        et_clf = joblib.load(f)
    
    return et_clf

def input_fn(request_body, content_type=JSON_CONTENT_TYPE):
    print(request_body, type(request_body))
    # request = json.loads(request_body)
    # url = request['url']
    data = json.loads(request_body)
    print(data, type(data))
    data_dataframe = pd.DataFrame.from_dict(data, orient="index").T
    
    # Preprocess
    le = LabelEncoder()
    categorical_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    def encode_categorical_feature(df, le):
        for feature in categorical_features:
            df[feature] = le.fit_transform(df[feature])
            
        return df
    data = encode_categorical_feature(data_dataframe, le)

    return data

def predict_fn(data, et_clf):
    preds = et_clf.predict(data)
    
    return preds
