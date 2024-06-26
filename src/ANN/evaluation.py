import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import save_plot

class Evaluator:
    def __init__(self, model):
        self.model = model

    def plot_loss(self, history):
        loss = history.history['loss']
        fig = plt.figure()
        sns.lineplot(x=range(len(loss)), y=loss)
        plt.title("Training Loss per Epoch")
        save_plot(fig, "training_loss.png")

    def evaluate(self, X_train, y_train, X_test, y_test):
        training_score = self.model.evaluate(X_train, y_train)
        test_score = self.model.evaluate(X_test, y_test)
        return training_score, test_score

    def plot_predictions(self, y_test, predictions):
        pred_df = pd.DataFrame(y_test, columns=['Test Y'])
        test_predictions = pd.Series(predictions.reshape(-1,))
        pred_df = pd.concat([pred_df, test_predictions], axis=1)
        pred_df.columns = ['Test Y', 'Model Predictions']
        
        fig = plt.figure()
        sns.scatterplot(x='Test Y', y='Model Predictions', data=pred_df)
        pred_df['Error'] = pred_df['Test Y'] - pred_df['Model Predictions']
        sns.histplot(pred_df['Error'], bins=50)
        save_plot(fig, "predictions_plot.png")

    def calculate_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        return mae, mse, rmse
