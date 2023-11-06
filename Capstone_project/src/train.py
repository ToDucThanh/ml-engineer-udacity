import os
import argparse
import logging
import sys
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import ExtraTreesClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Pipeline:
    """
    """
    def __init__(self, args):
        self.data_dir = args.data
        
        self.train_data_path = os.path.join(self.data_dir, "train.csv")
        self.test_data_path = os.path.join(self.data_dir, "test.csv")
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.model_save_path = args.model_output_dir
        
    def load_data(self):
        """Load train/test csv data"""
        
        try:
            logger.info("Load dataset ...")
            # Load train data
            train_data = pd.read_csv(self.train_data_path)
            self.X_train = train_data.iloc[:, :-1]
            self.y_train = train_data["HeartDisease"]
            
            # Load test data
            test_data = pd.read_csv(self.test_data_path)
            self.X_test = test_data.iloc[:, :-1]
            self.y_test = test_data["HeartDisease"]
            logger.info("Load dataset successfully!")          
        except Exception as e:
            logger.error(f"Error: {e}")
            fName = load_data.__name__
            raise Exception(f"Error in {fName} function")
            
    def fit_classifier(self):
        """Fit an extra tree classifier"""
        
        try:
            if self.X_train is None:
                self.load_data()
                
            logger.info("Create an extra tree classifier ...")
            et = ExtraTreesClassifier(
                bootstrap=False, 
                ccp_alpha=0.0, 
                class_weight=None,
                criterion='gini', 
                max_depth=None, 
                max_features='sqrt',
                max_leaf_nodes=None, 
                max_samples=None,
                min_impurity_decrease=0.0, 
                min_samples_leaf=1,
                min_samples_split=2, 
                min_weight_fraction_leaf=0.0,
                n_estimators=100, 
                n_jobs=-1, 
                oob_score=False,
                random_state=125, 
                verbose=0, 
                warm_start=False
            )
            logger.info("Fit the classifier ...")
            et.fit(self.X_train, self.y_train)
            
            return et  
        except Exception as e:
            logger.error(f"Error: {e}")
            fName = fit_classifier.__name__
            raise Exception(f"Error in {fName} function")            
            
    def save_model(self, classifier):
        try:
            logger.info("Save model ...")
            joblib.dump(classifier, os.path.join(self.model_save_path, "et_clf.joblib"))
            logger.info("Save model successfully!")
        except Exception as e:
            logger.error(f"Error: {e}")
            fName = save_model.__name__
            raise Exception(f"Error in {fName} function")            

def main(args):
    pipeline = Pipeline(args)
    # Load dataset
    pipeline.load_data()
    # Fit a classifier
    classifier = pipeline.fit_classifier()
    # Save the classifier
    pipeline.save_model(classifier)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Heart Disease Classification")
    parser.add_argument(
        "--model-output-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"]
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.environ["SM_CHANNEL_TRAINING"]
    )
    args = parser.parse_args()

    main(args)