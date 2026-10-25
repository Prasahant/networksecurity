import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig



from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score  

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
  AdaBoostClassifier,
  GradientBoostingClassifier,
  RandomForestClassifier
)
import mlflow

class ModelTrainer:
  def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
    try:
      self.model_trainer_config = model_trainer_config
      self.data_transformation_artifact = data_transformation_artifact
    except Exception as e:
      raise NetworkSecurityException(e, sys)
    
  
  
  def track_mlflow(
    self,
    best_model,
    train_metric,
    test_metric,
    model_name
  ):
    # Set the same tracking database used by MLflow UI
    db_path = r"F:\Network Security\mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.replace(chr(92), '/')}")

    # Create or select your project experiment
    mlflow.set_experiment("Network Security")
    with mlflow.start_run():
      mlflow.log_param("model_name", model_name)

      mlflow.log_metric(
        "train_f1_score",
        train_metric.f1_score
      )
      mlflow.log_metric(
        "train_precision",
        train_metric.precision_score
      )
      mlflow.log_metric(
        "train_recall",
        train_metric.recall_score
      )

      mlflow.log_metric(
        "test_f1_score",
        test_metric.f1_score
      )
      mlflow.log_metric(
        "test_precision",
        test_metric.precision_score
      )
      mlflow.log_metric(
        "test_recall",
        test_metric.recall_score
      )

      mlflow.sklearn.log_model(
        sk_model=best_model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
      )


  def train_model(self,x_train,y_train,x_test, y_test):
    models = {
        "Random Forest": RandomForestClassifier(verbose=1),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(verbose=1),
        "Logistic Regression": LogisticRegression(verbose=1),
        "Adaboost": AdaBoostClassifier()
    }
    params = {
      "Decision Tree": {
        'criterion': ['gini', 'entropy', 'log_loss'],
        # 'splitter': ['best', 'random'],
        # 'max_features': ['sqrt', 'log2'],
      },
      "Random Forest": {
        # 'criterion': ['gini', 'entropy', 'log_loss'],
        # 'max_features': ['sqrt', 'log2', None],
        'n_estimators': [8, 16, 32, 64, 128, 256]
      },
      "Gradient Boosting": {
        #'loss': ['log_loss', 'exponential'],
        'learning_rate': [.1, .01, .05, .001],
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        # 'criterion': ['squared_error', 'friedman_mse'],
        # 'max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators': [8, 16, 32, 64, 128, 256]
      },
      "Logistic Regression": {
        # 'penalty': ['l2'],
        # 'solver': ['lbfgs', 'liblinear'],
        # 'C': [0.1, 1, 10]
      },
      "Adaboost": {
        'learning_rate': [.1, .01, 0.5, .001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
        }
      #, "KNeighborsClassifier": {
      #   'n_neighbors': [3, 5, 7, 9],
      #   'weights': ['uniform', 'distance'],
      #   'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
      # }
    }
    model_report:dict=evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,models=models, params=params)
    ## To get best model score from dict
    best_model_name = max(
      model_report,
      key=model_report.get
    )

    best_model_score = model_report[best_model_name]

    best_model = models[best_model_name]

    logging.info(
        f"Best Model: {best_model_name}, "
        f"CV F1 Score: {best_model_score:.4f}"
    )
    # Predict on training data
    y_train_pred = best_model.predict(x_train)
    
    classification_train_metric=get_classification_score(y_true=y_train,y_pred= y_train_pred)
    
    #Track the eperiments with MLFlow
    #self.track_mlflow(best_model,classification_train_metric)
    
    # Predict on testing data
    y_test_pred = best_model.predict(x_test)
    classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
    
    #self.track_mlflow(best_model,classification_test_metric)
    # Log both metrics in one MLflow run
    self.track_mlflow(
      best_model=best_model,
      train_metric=classification_train_metric,
      test_metric=classification_test_metric,
      model_name=best_model_name
    )
    preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
    
    model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
    os.makedirs(model_dir_path, exist_ok=True)
    Network_model = NetworkModel(preprocessor=preprocessor, model = best_model)
    save_object(self.model_trainer_config.trained_model_file_path, obj=Network_model)
    
    #Model Trainer Artifact
    model_trainer_artifact= ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                        train_metric_artifact=classification_train_metric,
                        test_metric_artifact=classification_test_metric
                        )
    logging.info(f"Model trainer artifact: {model_trainer_artifact}")
    return model_trainer_artifact
    
    
    
  def initiate_model_trainer(self) -> ModelTrainerArtifact:
    try:
      logging.info("Entered initiate_model_trainer method of ModelTrainer class")
      
      train_file_path = self.data_transformation_artifact.transformed_train_file_path
      test_file_path = self.data_transformation_artifact.transformed_test_file_path

      # Loading training array and testing array
      train_arr = load_numpy_array_data(train_file_path)
      test_arr = load_numpy_array_data(test_file_path)
      
      x_train, y_train, x_test, y_test = (
        train_arr[:, :-1],
        train_arr[:, -1],
        test_arr[:, :-1],
        test_arr[:, -1]
      )
        
      model_trainer_artifact = self.train_model(x_train,y_train,x_test,y_test)
      return model_trainer_artifact
    except Exception as e:
      raise NetworkSecurityException(e,sys)