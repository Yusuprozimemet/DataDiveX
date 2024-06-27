import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.tree import DecisionTreeRegressor
from utils import get_config

class ANNModel:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        input_shape = self.config['model']['ann']['input_shape']
        for layer in self.config['model']['ann']['layers']:
            model.add(Dense(units=layer['units'], activation=layer['activation'], input_shape=(input_shape,)))
            input_shape = None  # Only set input shape for the first layer
        model.add(Dense(1))
        model.compile(optimizer=self.config['model']['ann']['optimizer'], loss=self.config['model']['ann']['loss'], metrics=self.config['model']['ann']['metrics'])
        return model

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size):
        history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
        return history

    def predict(self, X):
        return self.model.predict(X)

class DecisionTreeModel:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        return DecisionTreeRegressor(max_depth=self.config['model']['decision_tree']['max_depth'], random_state=self.config['model']['decision_tree']['random_state'])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

class HybridModel:
    def __init__(self, ann_model, tree_model):
        self.ann_model = ann_model
        self.tree_model = tree_model

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size):
        self.ann_model.train(X_train, y_train, X_test, y_test, epochs, batch_size)
        ann_predictions = self.ann_model.predict(X_train)
        self.tree_model.train(ann_predictions, y_train)

    def predict(self, X):
        ann_predictions = self.ann_model.predict(X)
        return self.tree_model.predict(ann_predictions)
