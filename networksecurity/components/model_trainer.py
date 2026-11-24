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
import mlflow.sklearn
import joblib
import dagshub




class ModelTrainer:
  def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
    try:
      self.model_trainer_config = model_trainer_config
      self.data_transformation_artifact = data_transformation_artifact
    except Exception as e:
      raise NetworkSecurityException(e, sys)
    
  
  
  
  
  
  def track_mlflow(
      self,
      models,
      train_metrics,
      test_metrics
  ):
    # Save model files locally
    local_artifact_dir = os.path.join(
        r"F:\Network Security",
        "Artifacts",
        "mlflow_models"
    )
    os.makedirs(local_artifact_dir, exist_ok=True)

    # Connect to DagsHub
    dagshub.init(
        repo_owner="Prasahant",
        repo_name="networksecurity",
        mlflow=True
    )

    mlflow.set_experiment("Network Security")

    # Log every model as a separate MLflow run
    for model_name, model in models.items():

      train_metric = train_metrics[model_name]
      test_metric = test_metrics[model_name]

      safe_name = model_name.replace(" ", "_")

      # Save model locally
      local_model_path = os.path.join(
          local_artifact_dir,
          f"{safe_name}.pkl"
      )
      joblib.dump(model, local_model_path)

      with mlflow.start_run(run_name=model_name):

          mlflow.log_param("model_name", model_name)

          mlflow.log_metrics({
              "train_f1_score": train_metric.f1_score,
              "train_precision": train_metric.precision_score,
              "train_recall": train_metric.recall_score,
              "test_f1_score": test_metric.f1_score,
              "test_precision": test_metric.precision_score,
              "test_recall": test_metric.recall_score
          })

          # Log trained model to DagsHub
          mlflow.sklearn.log_model(
              sk_model=model,
              name="model",
              skops_trusted_types=[
                  "sklearn.tree._tree.Tree"
              ]
          )

          # Upload the local model file to DagsHub
          mlflow.log_artifact(
              local_model_path,
              artifact_path="local_model_files"
          )
          logging.info(
                f"Model logged successfully: {model_name}"
            )
          logging.info(
              f"Local artifact saved at: {local_model_path}"
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
    model_report: dict = evaluate_models(
      x_train=x_train,
      y_train=y_train,
      x_test=x_test,
      y_test=y_test,
      models=models,
      params=params
    )

    # Get the best model
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

    # Calculate training and testing metrics for all models
    train_metrics = {}
    test_metrics = {}

    for model_name, model in models.items():

      # Predict on training data
      y_train_pred = model.predict(x_train)

      classification_train_metric = get_classification_score(
          y_true=y_train,
          y_pred=y_train_pred
      )

      # Predict on testing data
      y_test_pred = model.predict(x_test)

      classification_test_metric = get_classification_score(
          y_true=y_test,
          y_pred=y_test_pred
      )

      # Store metrics for each model
      train_metrics[model_name] = classification_train_metric
      test_metrics[model_name] = classification_test_metric

      logging.info(
          f"{model_name} | "
          f"Train F1: {classification_train_metric.f1_score:.4f} | "
          f"Test F1: {classification_test_metric.f1_score:.4f}"
    )

    # Track all models and their metrics using MLflow
    self.track_mlflow(
        models=models,
        train_metrics=train_metrics,
        test_metrics=test_metrics
    )

    # Get metrics for the best model
    classification_train_metric = train_metrics[best_model_name]
    classification_test_metric = test_metrics[best_model_name]

    # Load the preprocessing object
    preprocessor = load_object(
        file_path=self.data_transformation_artifact.transformed_object_file_path
    )

    # Create the local model directory
    model_dir_path = os.path.dirname(
        self.model_trainer_config.trained_model_file_path
    )

    os.makedirs(model_dir_path, exist_ok=True)

    # Create the final model using the best model
    network_model = NetworkModel(
        preprocessor=preprocessor,
        model=best_model
    )

    # Save the final model locally
    save_object(
        self.model_trainer_config.trained_model_file_path,
        obj=network_model
    )
    
    save_object("final_model/model.pkl",best_model)

    # Create the model trainer artifact
    model_trainer_artifact = ModelTrainerArtifact(
        trained_model_file_path=self.model_trainer_config.trained_model_file_path,
        train_metric_artifact=classification_train_metric,
        test_metric_artifact=classification_test_metric
    )

    logging.info(
        f"Model trainer artifact: {model_trainer_artifact}"
    )

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