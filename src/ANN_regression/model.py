import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.tree import DecisionTreeRegressor

class ANNModel:
    def __init__(self, config):
        self.config = config['model']['ann']
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        input_dim = self.config['input_dim']
        for layer_config in self.config['layers']:
            model.add(Dense(units=layer_config['units'], activation=layer_config['activation'], input_shape=(input_dim,)))
            input_dim = None  # Only set input shape for the first layer
        model.add(Dense(1))
        model.compile(optimizer=self.config['optimizer'], loss=self.config['loss'], metrics=self.config['metrics'])
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
