# multivariate multi-step encoder-decoder lstm
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import tensorflow as tf

# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks
    train, test = data[1:-224], data[-224:-7]
    # print(train)
    # restructure into windows of weekly data
    train = array(split(train, len(train) / 7))
    test = array(split(test, len(test) / 7))
    return train, test


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        # print('actual[:, i]',actual[:, i])
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        mape = mean_absolute_percentage_error(actual[:, i], predicted[:, i])
        r2 = r2_score(actual[:, i], predicted[:, i])
        # store
        scores.append(rmse)
    print('rmse',rmse)
    print('mse',mse)
    print('mape', mape)
    print('r2',r2)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y, z = list(), list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            X.append(data[in_start:in_end, :-1])
            y.append(data[in_end:out_end, -1])
            z.append(data[in_start:in_end, :])
        # move along one time step
        in_start += 1
    return array(X), array(y), array(z)


# train the model
def build_model(train, n_input):
    # prepare data
    train_x, train_y, train_all = to_supervised(train, n_input)
    # define parameters
    verbose, epochs, batch_size = 0, 50, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    train_y = train_y.reshape(train_y.shape[0] * train_y.shape[1], train_y.shape[2])
    w = train
    for i in range(5):
        train_x, train_y, train_all = to_supervised(w, n_input)
        predict_x = model.predict(train_x, verbose=0)
        train_y = train_y.reshape(train_y.shape[0], train_y.shape[1] , 1)
    
        train_y = tf.convert_to_tensor(train_y,dtype=tf.float32)
        predict_x = tf.convert_to_tensor(predict_x, dtype=tf.float32)
        tf.disable_v2_behavior()
        loss_op = tf.reduce_mean(tf.square(predict_x - train_y))
        raw_perturb = tf.gradients(loss_op, train_y)
        alpha = 0.001
        normalized_per = tf.nn.l2_normalize(raw_perturb)
        perturb= alpha * tf.sqrt(tf.cast(tf.shape(train_y)[0], tf.float32)) * tf.stop_gradient(normalized_per)  # 干
        w= train_all + perturb  # 干扰版输入
        w = tf.Session().run(w)
        w = w.reshape(w.shape[0] * w.shape[1], w.shape[2], w.shape[3])
    return model


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :-1]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    print(yhat)
    return yhat


# evaluate a single model
def evaluate_model(train, test, n_input):
    # fit model
    model = build_model(train, n_input)
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :, -1], predictions)
    return score, scores

import numpy as np
# load the new file
dataset = read_csv('testdata-TP.csv',encoding='gbk',header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'],dtype=np.float)
# split into train and test
w = [0.049,0.097,0.231,0.053,0.031,0.129,0.409,1]
data = dataset.values
for i in range(data.shape[0]):
    data[i,:] = np.multiply(data[i,:], w)
train, test = split_dataset(data)
# print(train,test)
# evaluate model and get scores
n_input = 7
score, scores = evaluate_model(train, test, n_input)
# summarize scores
summarize_scores('lstm', score, scores)
# plot scores
# days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
# pyplot.plot(days, scores, marker='o', label='lstm')
# pyplot.show()