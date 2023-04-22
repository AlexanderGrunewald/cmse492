import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential


def plot_distributions(X, y):
    """
    Plots the distributions of the element abundances and target values.

    Parameters:
        X (numpy.ndarray): A 2D array of shape (n_samples, n_features) containing the element abundances.
        y (numpy.ndarray): A 2D array of shape (n_samples, n_targets) containing the target values.

    Returns:
        None
    """
    sns.set_style('ticks')
    sns.set_palette('colorblind')

    # Plot the distributions of the abundances
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('Distribution of Element Abundances')
    for i, axis in enumerate(ax.flatten()):
        if i < X.shape[1]:
            sns.histplot(X[:, i], ax=axis)
            axis.set_xlabel('Abundance {}'.format(i+1))
    plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.4)
    plt.show()

    # Plot the distributions of the target values
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Distribution of Target Values')
    ax[0].set_xlabel('log10(t_rad)')
    ax[1].set_xlabel('w')
    sns.histplot(y[:, :20], ax=ax[0])
    sns.histplot(y[:, 20:], ax=ax[1])
    plt.subplots_adjust(top=0.85, wspace=0.4)
    plt.show()


def scale_and_split(X, y):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X = np.log10(X)
    y_log = np.log10(y[:, :20])
    y = np.concatenate((y_log, y[:, 20:]), axis=1)
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler_x, scaler_y


def build_model(X, layers, num_nodes, plot_model=False):
    model = Sequential()
    model.add(Dense(200, input_dim=X.shape[1], activation='softplus', kernel_initializer='he_normal', ))
    model.add(BatchNormalization(axis=-1))
    model.add(Dense(200, activation='softplus', kernel_initializer='VarianceScaling', ))
    model.add(BatchNormalization())
    model.add(Dense(40, activation='softplus', kernel_initializer='VarianceScaling', ))
    model.compile(loss='mean_squared_error', optimizer='nadam')
    if plot_model:
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
        plt.show()

    return model


def build_custom_model(X, y,layer_sizes=None, plot_model=False, actiavtion = 'softplus', optimizer = 'nadam', kernel_initializer = 'he_normal'):
    model = Sequential()
    model.add(Dense(layer_sizes[0], input_dim=X.shape[1], activation=actiavtion, kernel_initializer=kernel_initializer))
    model.add(BatchNormalization())

    for layer_size in layer_sizes[1:]:
        model.add(Dense(layer_size, activation=actiavtion, kernel_initializer='VarianceScaling'))
        model.add(BatchNormalization())
    model.add(Dense(y.shape[1], activation='linear', kernel_initializer='VarianceScaling'))
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    if plot_model:
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
        plt.show()

    return model


def train_model(model, X_train, y_train, X_test, y_test, plot_loss=False, epochs = 100, batch_size = 100):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
    if plot_loss:
        # plot loss over epochs
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(["train", "validation"], loc="upper left")
        plt.show()
        plt.savefig("train_model_val.jpg");
    return model


def plot_predictions(model, X_test, y_test, index=0):
    sns.set_style('ticks')
    sns.set_palette('colorblind')
    y_pred = model.predict(X_test)
    #y_pred = np.power(10, y_pred)
    #y_test = np.power(10, y_test)
    v_inner = np.array([1.100e+09, 1.145e+09, 1.190e+09, 1.235e+09, 1.280e+09, 1.325e+09,
                        1.370e+09, 1.415e+09, 1.460e+09, 1.505e+09, 1.550e+09, 1.595e+09,
                        1.640e+09, 1.685e+09, 1.730e+09, 1.775e+09, 1.820e+09, 1.865e+09,
                        1.910e+09, 1.955e+09])
    # 2 plots, 1 row, 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(v_inner, y_test[index, :20], label='Ground Truth')
    axs[0].plot(v_inner, y_pred[index, :20], label='Predicted', linestyle='--', color='red')
    axs[0].set_xlabel('Inner Velocity [m/s]')
    axs[0].set_ylabel('Radiative Temperature [K]')
    axs[0].legend()
    axs[1].plot(v_inner, y_test[index, 20:], label='Ground Truth')
    axs[1].plot(v_inner, y_pred[index, 20:], label='Predicted', linestyle='--', color='red')
    axs[1].set_xlabel('Inner Velocity [m/s]')
    axs[1].set_ylabel('Dillution Factor')
    axs[1].legend()
    plt.show()
    