import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

import tensorflow

from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM
from keras.losses import MeanSquaredError, MeanAbsoluteError, Huber, LogCosh
from keras.optimizers import Adam

from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch

class LSTMTuner(HyperModel):
    """
    Hyperparameter tuning of LSTM models using Keras Tuner.

    Initialize the Tuner with configurable ranges and options for hyperparameters.
    Parameters:
    - input_shape (tuple): Shape of the input data.
    - n_outputs (int): Number of output neurons.
    - layer_range (tuple): Min and max range of LSTM layers.
    - unit_range (tuple): Range of units in each LSTM layer.
    - activation_choices (list): List of activation functions to choose from.
    - learning_rate_range (tuple): Min and max range for the learning rate.
    - loss_functions (list): List of possible loss functions.
    - batch_size_range (tuple): Range for batch size selection.
    """
    def __init__(self, input_shape, n_outputs, layer_range, unit_range, activation_choices, learning_rate_range, loss_functions, batch_size_range):
        self.input_shape            = input_shape
        self.n_outputs              = n_outputs
        self.layer_range            = layer_range
        self.unit_range             = unit_range
        self.activation_choices     = activation_choices
        self.learning_rate_range    = learning_rate_range
        self.loss_functions         = loss_functions
        self.batch_size_range       = batch_size_range


    def build(self, hp):
        """
        Build and compile an LSTM model using hyperparameters defined by Keras Tuner.

        Parameters:
        - hp: Hyperparameters object provided by Keras Tuner.

        Returns:
        - model: Compiled LSTM model with selected hyperparameters.
        """
        model = Sequential()

        # Add LSTM layers based on hyperparameters
        num_layers = hp.Int('num_layers', self.layer_range[0], self.layer_range[1])
        for i in range(num_layers):
            if i == 0:
                model.add(LSTM(
                    units = hp.Int('units_0', min_value=self.unit_range[0], max_value=self.unit_range[1], step=10),
                    activation = hp.Choice('activation_0', values=self.activation_choices),
                    input_shape = self.input_shape,
                    return_sequences = num_layers > 1
                ))
            else:
                model.add(LSTM(
                    units = hp.Int(f'units_{i}', min_value=self.unit_range[0], max_value=self.unit_range[1], step=10),
                    activation = hp.Choice(f'activation_{i}', values=self.activation_choices),
                    return_sequences = i < num_layers - 1
                ))

        model.add(Dense(self.n_outputs))# Output layer

        # Selection of the loss function based on hyperparameters
        loss_function = hp.Choice('loss_function', values=self.loss_functions)
        if loss_function == 'MAE':
            loss = MeanAbsoluteError()
        elif loss_function == 'MSE':
            loss = MeanSquaredError()
        elif loss_function == 'Huber':
            loss = Huber()
        elif loss_function == 'LogCosh':
            loss = LogCosh()
        else:
            raise ValueError("Invalid loss function")

        # Choose batch size from the hyperparameters
        batch_size = hp.Int('batch_size', min_value=self.batch_size_range[0], max_value=self.batch_size_range[1], step=8)

        model.compile(
            optimizer=Adam(hp.Float('learning_rate', min_value=self.learning_rate_range[0], max_value=self.learning_rate_range[1], sampling='log')),
            loss=loss
        )
        return model
#%%
def prepare_data(data, n_lag=1, n_ahead=1, target_index=0, test_size=0.2):
    """
    Format time series data into input and output matrices for modeling.

    Parameters:
    - data (np.array): Time series data as a 2D Numpy array.
    - n_lag (int): Number of lagged observations as input (X).
    - n_ahead (int): Number of observations ahead to predict (Y).
    - target_index (int): Index of the target variable in the data.
    - test_size (float): Proportion of the dataset to allocate for the test set.

    Returns:
    - X_train, Y_train: Arrays of input and output for the training set.
    - X_val, Y_val: Arrays of input and output for the test set.
    """

    n_ft = data.shape[1]  # Number of features in the data

    X, Y = [], []  # Initialize input and output lists

    # Generate input-output pairs for the data
    for i in range(len(data) - n_lag - n_ahead):
        Y.append(data[(i + n_lag):(i + n_lag + n_ahead), target_index])
        X.append(data[i:(i + n_lag)])

    X, Y = np.array(X), np.array(Y)  # Convert lists to Numpy arrays

    # Reshape X for RNN input: [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], n_lag, n_ft))

    # Splitting into train and test sets
    X_train, Y_train = X[0:int(X.shape[0] * (1 - test_size))], Y[0:int(X.shape[0] * (1 - test_size))]
    X_val, Y_val = X[int(X.shape[0] * (1 - test_size)):], Y[int(X.shape[0] * (1 - test_size)):]

    return X_train, Y_train, X_val, Y_val
#%%
def scale_data(data, test_size):
    """
    Scales the data by normalizing it based on the training set's mean and standard deviation.

    Parameters:
    - data (pd.DataFrame): The dataset to be scaled.
    - test_size (float): The proportion of the dataset to be used as the test set.

    Returns:
    - pd.DataFrame: A concatenated DataFrame containing the scaled training and test sets.
    """
    rows = data.shape[0]

    # Split into train and test sets
    train = data[0:int(rows * (1 - test_size))]
    test = data[int(rows * (1 - test_size)):]

    # Scale the data
    train_mean = train.mean()
    train_std = train.std()

    train = (train - train_mean) / train_std
    test = (test - train_mean) / train_std

    return pd.concat([train, test]), train_mean, train_std
#%%
def calculate_residuals(Y_val, forecast, train_std, train_mean, variable_name, N=7):
    """
    Function to calculate the residuals of predicting the validation set for each day ahead of prediction.

    Args:
    Y_val (numpy.ndarray): The actual values of the validation set.
    forecast (numpy.ndarray): The predicted values for the validation set.
    train_std (dict): The standard deviation of the training set, used for scaling.
    train_mean (dict): The mean of the training set, used for scaling.
    variable_name (str): The name of the variable for which the prediction is made.
    N (int): The number of days ahead of prediction

    Returns:
    pandas.DataFrame: DataFrame where each column is the residual for each day ahead of prediction.
    """
    # Initialize a list to store residuals for each day ahead
    daily_residuals = {f"{variable_name}_ahead{i+1}": [] for i in range(N)}

    # Iterate over each forecast and true value pair in the validation set
    for i in range(Y_val.shape[0]):
        true = Y_val[i]
        hat = forecast[i]

        # Scale back to original values
        true_scaled = np.asarray([(x * train_std[variable_name]) + train_mean[variable_name] for x in true])
        hat_scaled = np.asarray([(x * train_std[variable_name]) + train_mean[variable_name] for x in hat])

        # Calculate residuals for each day
        residuals = true_scaled - hat_scaled

        # Append residuals to respective lists
        for j, residual in enumerate(residuals):
            daily_residuals[f"{variable_name}_ahead{j+1}"].append(residual)

    # Convert the dictionary of lists into a DataFrame
    residuals_df = pd.DataFrame(daily_residuals)

    return residuals_df
#%%
def plot_error_density(residuals,feature_title,feature_name, feature_unit, n_ahead, save=False):
    if save:
        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists(f'results/N={n_ahead}'):
            os.makedirs(f'results/N={n_ahead}')
        filename = f'results/N={n_ahead}/LSTM_Error Density_{feature_name}_N={n_ahead}.svg'

    plt.figure(figsize=(8, 6))
    for i, column in enumerate(residuals.columns):
        sns.kdeplot(residuals[column], label=f'Day {i+1}')
    plt.xlabel(f'{feature_title} Predictive Error ({feature_unit})')
    plt.ylabel('Density')
    plt.legend()
    if save:
        plt.savefig(filename, format='svg')
        print(f"Saved: {filename}")
    else:
        plt.show()
#%%
def check_gpu():
    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
    if tensorflow.config.list_physical_devices('GPU'):
        print("TensorFlow will run on GPU.")
    else:
        print("TensorFlow will run on CPU.")