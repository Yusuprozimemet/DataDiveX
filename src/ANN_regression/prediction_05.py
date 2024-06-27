import yaml
import pickle
import csv
from model import ANNModel
from prediction_utils import Predictor, prepare_input_data

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Path to configuration file
    config_path = 'config.yaml'

    # Load configuration from YAML
    config = load_config(config_path)

    # Extract paths and parameters from configuration
    model_h5_path = config['model_paths']['model_h5_path']
    scaler_pkl_path = config['model_paths']['scaler_pkl_path']
    input_data = config['prediction']['input_data']['new_gem']

    # Load scaler
    with open(scaler_pkl_path, 'rb') as file:
        scaler = pickle.load(file)

    # Initialize ANNModel with the 'ann' configuration
    ann_model = ANNModel(config)

    # Load model weights
    ann_model.model.load_weights(model_h5_path)

    # Prepare input data for prediction
    input_data_scaled = prepare_input_data(input_data, scaler)

    # Make prediction
    predictor = Predictor(ann_model, scaler)
    prediction = predictor.make_prediction(input_data_scaled)
    print(f"Prediction for new data: {prediction}")

    # Save prediction to CSV file
    save_prediction_to_csv(prediction)

def save_prediction_to_csv(prediction):
    # Define the path to save the CSV file
    prediction_csv_path = 'artifacts/prediction_result.csv'

    # Write prediction to CSV file
    with open(prediction_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Prediction'])
        csv_writer.writerow([prediction])

if __name__ == "__main__":
    main()
