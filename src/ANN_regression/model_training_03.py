import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from data_ingestion import DataIngestion
from data_preprocessing import DataPreprocessing
from model import ANNModel, DecisionTreeModel, HybridModel
from evaluation import Evaluator
from prediction_utils import Predictor
from utils import save_model, get_config

def save_evaluation_results(evaluation_results, model_config, filepath):
    """
    Save evaluation results and model configuration to a CSV file.
    
    Args:
    - evaluation_results (dict): Dictionary containing evaluation metrics.
    - model_config (dict): Dictionary containing model configuration.
    - filepath (str): Filepath to save the CSV file.
    """
    # Convert model_config to DataFrame
    model_config_df = pd.DataFrame([model_config])
    
    # Combine evaluation results and model configuration
    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_with_config_df = pd.concat([evaluation_df, model_config_df], axis=1)
    
    # Save to CSV
    evaluation_with_config_df.to_csv(filepath, index=False)
    print(f"Evaluation results and model configuration saved to {filepath}")

def main():
    # Reload updated config with columns
    config = get_config()

    # Load your preprocessed data (assuming it's already preprocessed and saved)
    X_train = pd.read_csv('artifacts/X_train.csv')
    X_test = pd.read_csv('artifacts/X_test.csv')
    y_train = pd.read_csv('artifacts/y_train.csv').values.ravel()
    y_test = pd.read_csv('artifacts/y_test.csv').values.ravel()

    # Load scaler
    scaler_filepath = 'artifacts/scaler.pkl'
    with open(scaler_filepath, 'rb') as file:
        scaler = pickle.load(file)

    # Select and Train Model
    model_type = config['model']['type']
    if model_type == "ANN":
        ann_model = ANNModel(config)
        history = ann_model.train(X_train, y_train, X_test, y_test,
                                  epochs=config['model']['ann']['epochs'],
                                  batch_size=config['model']['ann']['batch_size'])
        model = ann_model
    elif model_type == "ANN_with_DT":
        ann_model = ANNModel(config)
        tree_model = DecisionTreeModel(config)
        hybrid_model = HybridModel(ann_model, tree_model)
        hybrid_model.train(X_train, y_train, X_test, y_test,
                           epochs=config['model']['ann']['epochs'],
                           batch_size=config['model']['ann']['batch_size'])
        model = hybrid_model
    else:
        raise ValueError(f"Model type {model_type} not recognized.")

    # Evaluate Model
    evaluator = Evaluator(model)
    mae, mse, rmse, evs = evaluator.evaluate_performance(y_test, model.predict(X_test))
    evaluation_results = {
        'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'MAE': [mae],
        'MSE': [mse],
        'RMSE': [rmse],
        'EVS': [evs]
    }
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, EVS: {evs}")

    # Add model type to the configuration for saving evaluation results
    config['model'][model_type.lower()]['type'] = model_type

    # Save Evaluation Results with Model Configuration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    evaluation_filepath = f"artifacts/evaluation_results_{timestamp}.csv"
    save_evaluation_results(evaluation_results, config['model'][model_type.lower()], evaluation_filepath)

    # Save Model
    model_filepath = f"models/model_{timestamp}.h5"  # Adjust file extension based on model type
    save_model(model.ann_model.model if model_type == "ANN_with_DT" else model.model, model_filepath)

if __name__ == "__main__":
    main()
