import matplotlib.pyplot as plt
import seaborn as sns
from utils import save_plot

class EDA:
    def plot_histograms(self, df):
        num_columns = len(df.columns)
        num_plots = min(num_columns, 20)  # Limit to 20 plots for visualization
        fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 15))  # Adjust based on the number of columns
        axes = axes.flatten()
        
        for i, col in enumerate(df.columns):
            if i >= num_plots:
                break
            if df[col].dtype in ['float64', 'int64']:  # Only plot numeric columns
                df[col].hist(bins=30, ax=axes[i])
                axes[i].set_title(col)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])  # Remove unused subplots
        
        save_plot(fig, 'histograms.png')

    def plot_scatter(self, df):
        plt.figure(figsize=(12, 8))
        scatter_fig = sns.scatterplot(x='price', y='sqft_living', data=df).get_figure()
        save_plot(scatter_fig, 'scatter_price_vs_sqft_living.png')

    def plot_boxplots(self, df):
        fig, axes = plt.subplots(3, 2, figsize=(14, 21))
        sns.boxplot(x='bedrooms', y='price', data=df, ax=axes[0, 0])
        sns.boxplot(x='floors', y='price', data=df, ax=axes[0, 1])
        sns.boxplot(x='bathrooms', y='price', data=df, ax=axes[1, 0])
        sns.boxplot(x='grade', y='price', data=df, ax=axes[1, 1])
        sns.boxplot(x='condition', y='price', data=df, ax=axes[2, 0])
        fig.delaxes(axes[2, 1])
        save_plot(fig, 'boxplots.png')

    def plot_heatmap(self, df):
        plt.figure(figsize=(20, 15))
        ax = plt.gca()
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        heatmap_fig = sns.heatmap(numeric_df.corr(), annot=True, ax=ax).get_figure()
        save_plot(heatmap_fig, 'heatmap.png')

    def plot_scatter_location(self, df):
        plt.figure(figsize=(12, 8))
        scatter_loc_fig = sns.scatterplot(x='long', y='lat', hue='price', data=df, palette='viridis').get_figure()
        save_plot(scatter_loc_fig, 'scatter_location.png')

    def plot_price_vs_house_age(self, df):
        plt.figure(figsize=(12, 8))
        line_fig = sns.lineplot(x='house_age', y='price', data=df).get_figure()
        save_plot(line_fig, 'lineplot_price_vs_house_age.png')
