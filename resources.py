import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_data() -> pd.DataFrame:
    """
    Load the Covertype dataset from the UCI Machine Learning repository.

    Returns:
    --------
    A pandas DataFrame containing the dataset.
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
    col_names = [
        'Elevation', 'Aspect', 'Slope',
        'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points'
    ]
    wilderness_area = ['Wilderness_Area' + str(i) for i in range(1, 5)]
    soil_type = ['Soil_Type' + str(i) for i in range(1, 41)]
    col_names += wilderness_area + soil_type + ['Cover_Type']

    df = pd.read_csv(url, compression='gzip', header=None, names=col_names)
    return df

def split_data(df :pd.DataFrame, test_size=0.1, validation_size=0.1, seed=42) -> tuple(pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    """Split data into train, validation, and test sets.

    Args:
        df (pandas.DataFrame): Input data.
        test_size (float): Proportion of data to be used for test set. Default is 0.1.
        validation_size (float): Proportion of data to be used for validation set. Default is 0.1.
        seed (int): Random seed for train-test split. Default is 42.

    Returns:
        tuple: X_train (pandas.DataFrame), y_train (pandas.Series), X_valid (pandas.DataFrame), y_valid (pandas.Series), X_test (pandas.DataFrame), y_test (pandas.Series).
    """
    # Split data into features (X) and target variable (y)
    X = df.drop('Cover_Type', axis=1)
    y = df['Cover_Type']

    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=validation_size, random_state=seed)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def train_one_r(X_train: pd.DataFrame, y_train: pd.Series, column="Elevation") -> DecisionTreeClassifier:
    """Create a decision tree with a specified column as the predictor variable
    and fit it to the training data.

    Args:
        X_train (pandas.DataFrame): Training data features.
        y_train (pandas.Series): Training data target.
        column (str): Name of the column to use as predictor variable. Default is "Elevation".

    Returns:
        DecisionTreeClassifier: Fitted decision tree.
    """
    # Create decision tree classifier
    tree = DecisionTreeClassifier(max_depth=1)

    # Fit decision tree to training data
    tree.fit(X_train[[column]], y_train)

    return tree

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, n_estimators=50, min_samples_leaf=5, max_features=0.5, n_jobs=-1) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier on the input data.

    Args:
        X_train (pandas.DataFrame): Training data features.
        y_train (pandas.Series): Training data target.
        n_estimators (int): Number of trees in the forest. Default is 50.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node. Default is 5.
        max_features (float): Maximum number of features to consider when looking for the best split. Default is 0.5.
        n_jobs (int): Number of jobs to run in parallel for both fit and predict. Default is -1.

    Returns:
        RandomForestClassifier: Trained random forest classifier.
    """
    # Create random forest classifier
    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=n_jobs)

    # Fit random forest to training data
    rf.fit(X_train, y_train)

    return rf

def train_knn(X_train: pd.DataFrame, y_train: pd.Series, n_neighbors=7, algorithm='ball_tree', n_jobs=-1) -> KNeighborsClassifier:
    """
    Train a K-Nearest Neighbors Classifier on the input data.

    Args:
        X_train (pandas.DataFrame): Training data features.
        y_train (pandas.Series): Training data target.
        n_neighbors (int): Number of neighbors to use. Default is 7.
        algorithm (str): Algorithm used to compute the nearest neighbors. Default is 'ball_tree'.
        n_jobs (int): Number of parallel jobs to run for neighbors search. Default is -1.

    Returns:
        KNeighborsClassifier: Trained K-Nearest Neighbors classifier.
    """
    # Create K-Nearest Neighbors classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm, n_jobs=n_jobs)

    # Fit K-Nearest Neighbors to training data
    knn.fit(X_train, y_train)

    return knn

def create_nn(n_layers=2, n_neurons=64, lr=0.001, dropout_rate=0.2) -> Sequential:
    """
    Creates a neural network model with specified parameters.

    Args:
        n_layers (int): Number of hidden layers in the model. Default is 2.
        n_neurons (int): Number of neurons in each hidden layer. Default is 64.
        lr (float): Learning rate of the optimizer. Default is 0.001.
        dropout_rate (float): Dropout rate for regularization. Default is 0.2.

    Returns:
        tensorflow.keras.models.Sequential: Compiled neural network model.
    """
    model = Sequential()

    # Input Layer
    model.add(Dense(n_neurons, activation='relu', input_dim=54))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Hidden Layers
    for i in range(n_layers-1):
        model.add(Dense(n_neurons, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    # Output Layer    
    model.add(Dense(7, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, nn=False) -> tuple(float, np.ndarray):
    """Evaluates a trained model on test data and returns the accuracy and predicted values.

    Args:
        model: Trained model object with a `predict` method.
        X_test (array-like): Test data features.
        y_test (array-like): Test data target.
        nn (bool): If True, adds 1 to the predicted values before computing accuracy. Default is False.

    Returns:
        tuple: Model accuracy and predicted values on the test data.
    """
    y_pred = model.predict(X_test)
    if nn:
        y_pred = np.argmax(y_pred, axis=1) + 1
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

def plot_history(history: tf.keras.callbacks.History) -> None:
    """
    Plot the training and validation accuracy and loss of a model.

    Args:
        history: The history of the model.
    """
    # Plot the training and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()

    # Plot the training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()