import pandas as pd

def compare_evaluation_results(evaluation_files):
    best_model = None
    best_metrics = None

    for file in evaluation_files:
        print(f"Processing file: {file}")
        if file.endswith('.csv') and 'evaluation_results' in file:
            try:
                # Read CSV file
                df = pd.read_csv(file)
                if df.empty:
                    print(f"File {file} is empty.")
                    continue
                
                # Extract model configuration from the first row (assuming it's the same for all entries in the file)
                model_config = df.iloc[0].to_dict()
                print(f"Model configuration: {model_config}")

                # Extract evaluation metrics
                evaluation_metrics = df[['MAE', 'MSE', 'RMSE', 'EVS']].mean().to_dict()
                print(f"Evaluation metrics: {evaluation_metrics}")

                # Compare metrics to determine the best model
                if best_metrics is None or evaluation_metrics['MSE'] < best_metrics['MSE']:
                    best_model = model_config
                    best_metrics = evaluation_metrics
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    
    return best_model, best_metrics

def save_results_to_csv(best_model, best_metrics, output_file):
    # Create a DataFrame from best_model and best_metrics
    results_dict = {**best_model, **best_metrics}
    results_df = pd.DataFrame([results_dict])

    # Save to CSV
    results_df.to_csv(output_file, index=False)
