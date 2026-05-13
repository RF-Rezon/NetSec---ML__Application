import os
import sys

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

# from main utils
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models

# from ml utils
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC

import mlflow

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def track_mlflow(self, best_model, train_metric, test_metric):
        # একটি সিঙ্গেল রানের মধ্যে সবকিছু লগ করা
        with mlflow.start_run():
            # লগিং প্যারামিটার (ঐচ্ছিক কিন্তু জরুরি)
            mlflow.log_param("model_name", type(best_model).__name__)       # এটি মডেলের নাম সেভ করে (যেমন: RandomForestClassifier)। যাতে পরে ড্যাশবোর্ডে দেখে বোঝা যায় কোন মডেলটি এই রেজাল্ট দিয়েছে।
            
            # ট্রেইন মেট্রিক্স লগ করা
            mlflow.log_metric("train_f1_score", train_metric.f1_score)
            mlflow.log_metric("train_precision", train_metric.precision_score)
            mlflow.log_metric("train_recall", train_metric.recall_score)
            
            # টেস্ট মেট্রিক্স লগ করা
            mlflow.log_metric("test_f1_score", test_metric.f1_score)
            mlflow.log_metric("test_precision", test_metric.precision_score)
            mlflow.log_metric("test_recall", test_metric.recall_score)
            
            # মডেল সেভ করা
            mlflow.sklearn.log_model(best_model, "model")

        
    def train_model(self,X_train,y_train,X_test,y_test):

        models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "XGBClassifier": XGBClassifier(random_state=42, eval_metric='logloss'),
        "CatBoost Classifier": CatBoostClassifier(random_state=42, verbose=False),
        "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "SVM": SVC(random_state=42, probability=True)
        }

        params = {
        "Decision Tree": {
            "max_depth": [3, 5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "max_features": [None, "sqrt", "log2"],
        },
        
        "Gradient Boosting": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "subsample": [0.8, 0.9, 1.0],
            "max_features": [None, "sqrt"],
        },
        
        "Logistic Regression": {
            "C": [0.01, 0.1, 1.0, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs", "saga"],
            "max_iter": [1000, 2000],
            "class_weight": [None, "balanced"],
            "l1_ratio": [0.1, 0.5, 0.9],  # for elasticnet
        },
        
        "XGBClassifier": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 9],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "gamma": [0, 0.1, 0.2],
            "reg_alpha": [0, 0.01, 0.1],
            "reg_lambda": [0.5, 1, 2],
        },
        
        "CatBoost Classifier": {
            "iterations": [100, 200, 300],
            "learning_rate": [0.03, 0.05, 0.1],
            "depth": [4, 6, 8],
            "l2_leaf_reg": [1, 3, 5],
            "min_data_in_leaf": [1, 3, 5],
            "bagging_temperature": [0, 0.5, 1],
        },
        
        "AdaBoost Classifier": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.05, 0.1, 0.5, 1.0]
        },
        
        "Random Forest": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [10, 15, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "criterion": ["gini", "entropy", "log_loss"],
            "bootstrap": [True, False],
            "class_weight": [None, "balanced"],
        },
        
        "KNeighborsClassifier": {
            "n_neighbors": [3, 5, 7, 11, 15, 21],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree"],
            "p": [1, 2, 3],
            "leaf_size": [20, 30, 40],
            "metric": ["euclidean", "manhattan", "chebyshev"],
        },
        
        "SVM": {
            "C": [0.01, 0.1, 1, 10, 100],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["scale", "auto", 0.01, 0.1, 1],
            "degree": [2, 3, 4],  # for poly kernel
            "coef0": [0, 0.1, 1],  # for poly/sigmoid
            "class_weight": [None, "balanced"],
            "shrinking": [True, False],
        }
        }
            
        
        # মডেল ইভ্যালুয়েশন
        model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                            models=models, param=params)

        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]
        
        # মডেল ফিট করা
        best_model.fit(X_train, y_train)

        # প্রেডিকশন এবং স্কোর ক্যালকুলেশন
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        # একসাথে MLflow-তে ডাটা পাঠানো
        self.track_mlflow(best_model, classification_train_metric, classification_test_metric)

        # ------------------------------------------------------------------------------------------------------------------


        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        network_model= NetworkModel(preprocessor=preprocessor,model=best_model)

        save_object(self.model_trainer_config.trained_model_file_path,obj= network_model)    # Object tao rakha thaklo [ preprocessor +  model ]


        # --------------------------------------------------------------------------------------------------------------------

        save_object("final_model/model.pkl",best_model)     # Final model er jonno. 

        # --------------------------------------------------------------------------------------------------------------------


        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact


    
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            mlflow.set_experiment("Network_Security_Project")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        