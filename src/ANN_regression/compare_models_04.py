import os
from compare_model_utils import compare_evaluation_results, save_results_to_csv

def main():
    # Directory where evaluation files are stored
    evaluation_dir = 'artifacts'
    
    # Ensure the evaluation directory exists
    if not os.path.exists(evaluation_dir):
        print(f"Directory {evaluation_dir} does not exist.")
        return

    # Get list of all evaluation files
    evaluation_files = [os.path.join(evaluation_dir, f) for f in os.listdir(evaluation_dir) if f.startswith('evaluation_results') and f.endswith('.csv')]
    
    # Print debug information
    print(f"Found {len(evaluation_files)} evaluation files.")
    for file in evaluation_files:
        print(f"Found evaluation file: {file}")
    
    # Compare evaluation results
    best_model, best_metrics = compare_evaluation_results(evaluation_files)
    
    # Print or save results
    if best_model is not None:
        print("Best Model Configuration:")
        print(best_model)
        print("Best Evaluation Metrics:")
        print(best_metrics)

        # Save results to CSV
        output_file = os.path.join(evaluation_dir, 'best_model_evaluation_results.csv')
        save_results_to_csv(best_model, best_metrics, output_file)
        print(f"Results saved to {output_file}")
    else:
        print("No evaluation files found in the directory.")

if __name__ == "__main__":
    main()
