import os
import matplotlib.pyplot as plt

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
