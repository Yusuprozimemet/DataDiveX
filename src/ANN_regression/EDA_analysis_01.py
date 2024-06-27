import pandas as pd
from data_ingestion import DataIngestion
from EDA import EDA
from utils import initial_update_config_with_columns, update_config_with_house_age, get_config

def calculate_house_age(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['house_age'] = df['year'] - df['yr_built']
    return df

try:
    # Load configuration
    config = get_config()
    print(config)  # Debug print to check the content of the config

    # Ingest data
    data_ingestion = DataIngestion(config)
    df = data_ingestion.load_data()

    if df is not None:
        # Initial update of config with columns
        initial_update_config_with_columns(df, 'config.yaml')

        # Exploratory Data Analysis (EDA) - Generate initial plots
        eda = EDA()
        eda.plot_histograms(df)
        eda.plot_scatter(df)
        eda.plot_boxplots(df)
        eda.plot_heatmap(df)
        eda.plot_scatter_location(df)

        # Calculate house_age
        df = calculate_house_age(df)

        # Update config with columns and house_age
        update_config_with_house_age(df, 'config.yaml')

        # Generate line plot of price vs house_age
        eda.plot_price_vs_house_age(df)
    else:
        print("Data loading process failed. Check error messages above.")

except KeyError as e:
    print(f"KeyError: {e} not found in the configuration.")
except Exception as e:
    print(f"Error: {str(e)}")
