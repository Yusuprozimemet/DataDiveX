import numpy as np
import yaml
import pandas as pd
from data_ingestion import DataIngestion
from data_preprocessing import DataPreprocessing
from EDA import EDA
from model import ANNModel, DecisionTreeModel, HybridModel
from evaluation import Evaluator
from prediction import Predictor
from utils import save_model, get_config

# Reload updated config with columns
config = get_config()

# Data Preprocessing
data_ingestion = DataIngestion(config)
df = data_ingestion.load_data()

data_preprocessing = DataPreprocessing(config)
X_train, X_test, y_train, y_test, scaler = data_preprocessing.preprocess_data(df)

# Select and Train Model
model_type = config['model']['type']
if model_type == "ANN":
    ann_model = ANNModel(config)
    history = ann_model.train(X_train, y_train, X_test, y_test, epochs=config['model']['ann']['epochs'], batch_size=config['model']['ann']['batch_size'])
    model = ann_model
elif model_type == "ANN_with_DT":
    ann_model = ANNModel(config)
    tree_model = DecisionTreeModel(config)
    hybrid_model = HybridModel(ann_model, tree_model)
    hybrid_model.train(X_train, y_train, X_test, y_test, epochs=config['model']['ann']['epochs'], batch_size=config['model']['ann']['batch_size'])
    model = hybrid_model
else:
    raise ValueError(f"Model type {model_type} not recognized.")

# Evaluate Model
evaluator = Evaluator(model)
evaluator.plot_loss(history)
mae, mse, rmse, evs = evaluator.evaluate_performance(y_test, model.predict(X_test))
print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, EVS: {evs}")

# Make Predictions
predictor = Predictor(model, scaler)
new_gem = np.array(config['prediction']['new_gem'])
prediction_result = predictor.make_prediction(new_gem)
print(f"Prediction for new data: {prediction_result}")

# Save Model
save_model(model.ann_model.model if model_type == "ANN_with_DT" else model.model, config['saved_model']['filepath'])
