import os
import matplotlib.pyplot as plt
from numpy import array
from sklearn.metrics import mean_squared_error
import math
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')


def machinelearningcode(stockticker):
    filename = f"{stockticker}.csv"

    # Read the CSV file from the local system
    df = pd.read_csv(filename)

    df1 = df['close']

    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    training_size = int(len(df1) * 0.65)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size,
                                :], df1[training_size:len(df1), :1]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=1, batch_size=64, verbose=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
    test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))
    print('Train RMSE:', train_rmse)
    print('Test RMSE:', test_rmse)

    look_back = 100
    trainPredictPlot = numpy.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(
        train_predict) + look_back, :] = train_predict

    testPredictPlot = numpy.empty_like(df1)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(train_predict) + (look_back * 2):len(df1) - 2, :] = test_predict

    len(test_data)
    x_input = test_data.reshape(1, -1)

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    lst_output = []
    n_steps = 100
    i = 0
    while i < 365:
        if len(temp_input) > n_steps:
            x_input = np.array(temp_input[-n_steps:])
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i += 1
        else:
            x_input = np.array(temp_input)
            x_input = x_input.reshape((1, len(temp_input), 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i += 1

    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 131)

    df3 = df1.tolist()
    df3.extend(lst_output)

    predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))
    predictions = predictions.flatten()
    actual_values = scaler.inverse_transform(
        test_data[-len(lst_output):]).flatten()

    percentage_error = np.abs(
        (predictions - actual_values) / actual_values) * 100
    print(percentage_error.tolist())
    accuracy_percentage = 100 - np.mean(percentage_error)
    print('Accuracy Percentage:', accuracy_percentage)

    return predictions.tolist(), test_rmse
