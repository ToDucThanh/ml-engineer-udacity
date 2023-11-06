import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def model_fn(model_dir):
    with open(os.path.join(model_dir, "et_clf.joblib"), "rb") as f:
        et_clf = joblib.load(f)
    
    return et_clf

def input_fn(request_body):
    data_dataFrame = pd.DataFrame(
        {k: v for k, v in request_body.dict().items()}
    )
    # Preprocess
    le = LabelEncoder()
    categorical_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    def encode_categorical_feature(df, le):
        for feature in categorical_features:
            df[feature] = le.fit_transform(df[feature])
            
        return df
    data = encode_categorical_feature(data_dataFrame, le)

    return data

def predict_fn(data, et_clf):
    preds = et_clf.predict(data)
    
    return preds.tolist()