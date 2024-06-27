import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from utils import save_plot

class Evaluator:
    def __init__(self, model):
        self.model = model

    def plot_loss(self, history):
        fig, ax = plt.subplots()
        pd.DataFrame(history.history).plot(ax=ax)
        save_plot(fig, 'loss_plot.png')

    def evaluate_performance(self, y_test, predictions):
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        evs = explained_variance_score(y_test, predictions)
        return mae, mse, rmse, evs
