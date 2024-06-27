import os
from matplotlib import pyplot as plt
import yaml

def save_config(config, config_path='config.yaml'):
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

# Your existing functions...

def save_plot(figure, filename, folder='artifacts'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)
    figure.savefig(filepath)
    plt.close(figure)
    print(f"Plot saved as {filepath}")

def save_model(model, filepath):
    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')
    model.save(os.path.join('artifacts', filepath))
    print(f"Model saved as artifacts/{filepath}")

def read_columns(filepath):
    df = pd.read_csv(filepath)
    return df.columns.tolist()

def initial_update_config_with_columns(df, config_path='config.yaml'):
    columns_to_drop = ['id', 'date', 'zipcode', 'price']
    columns = [col for col in df.columns if col not in columns_to_drop]

    # Read the existing config file
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = {}

    # Ensure the 'data' key exists in the config
    if 'data' not in config:
        config['data'] = {}

    # Preserve existing keys and update the necessary parts
    config['data']['columns'] = columns
    config['data']['independent_vars'] = columns  # Initial independent vars without house_age
    config['data']['dependent_var'] = 'price'  # Default dependent variable

    # Write the updated config back to the file
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    
    print(f"Initial update config file '{config_path}' with columns: {columns}")

def update_config_with_house_age(df, config_path='config.yaml'):
    columns_to_drop = ['id', 'date', 'zipcode', 'price']
    columns = [col for col in df.columns if col not in columns_to_drop]

    # Adding 'house_age' to the list of independent variables
    independent_vars = columns + ['house_age']

    # Read the existing config file
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = {}

    # Ensure the 'data' key exists in the config
    if 'data' not in config:
        config['data'] = {}

    # Update independent_vars only if it doesn't contain 'house_age'
    if 'house_age' not in config['data']['independent_vars']:
        config['data']['independent_vars'].append('house_age')

    # Write the updated config back to the file
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    
    print(f"Updated config file '{config_path}' with house_age.")

def get_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
