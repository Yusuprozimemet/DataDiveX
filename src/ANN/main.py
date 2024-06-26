# main.py
import numpy as np
import yaml
from data_processing import DataProcessor
from model import ANNModel
from evaluation import Evaluator
from prediction import Predictor
from utils import save_model

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load and preprocess data
data_processor = DataProcessor(config)
df = data_processor.load_data()
X_train, X_test, y_train, y_test, scaler = data_processor.preprocess_data(df)

# Build and train model
ann_model = ANNModel(config)
history = ann_model.train(X_train, y_train, epochs=config['model']['epochs'])

# Evaluate model
evaluator = Evaluator(ann_model)
evaluator.plot_loss(history)
training_score, test_score = evaluator.evaluate(X_train, y_train, X_test, y_test)
print(f"Training Score: {training_score}")
print(f"Test Score: {test_score}")

test_predictions = ann_model.predict(X_test)
evaluator.plot_predictions(y_test, test_predictions)
mae, mse, rmse = evaluator.calculate_metrics(y_test, test_predictions)
print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

# Make new predictions
new_gem = np.array(config['prediction']['new_gem'])
predictor = Predictor(ann_model, scaler)
prediction_result = predictor.make_prediction(new_gem)
print(f"Prediction for new data: {prediction_result}")

# Save plots and model
save_model(ann_model.model, config['saved_model']['filepath'])
