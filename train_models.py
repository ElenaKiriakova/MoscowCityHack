import numpy as np
import pandas as pd


from datetime import datetime

import datetime as dt
import time

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Dense, Flatten, Dropout, SimpleRNN, BatchNormalization, Bidirectional , GRU, LSTM

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

from keras.models import load_model
from keras.utils.vis_utils import plot_model


"""  FUNCTIONS  """


def time_2ms(str_time):
  # Функция преобразует столбец, содержащий время, в секунды с  1 января 1970 года
  # dt_end = datetime.strptime(str_time, "%Y-%m-%d %H:%M:%S")
  dt_start = datetime.strptime('1970-01-01 00:00:00', "%Y-%m-%d %H:%M:%S")
  dt_epoch = (str_time - dt_start).total_seconds()
  return dt_epoch


"""  LOAD AND PREPARE DATA """



data_or = pd.read_csv('data_hack.csv',
                      on_bad_lines='skip',
                      sep=';',
                      parse_dates=['create_date'],
                      infer_datetime_format=True
                      )


nn_data = data_or.copy()
nn_data = nn_data[nn_data['create_date'] >= '01-01-2017']
nn_data = nn_data.drop_duplicates()
nn_data = nn_data.dropna(how='any')
nn_data['avg_bill'] = nn_data['avg_bill']/70
nn_data.dropna(how='any', inplace=True)


"""  TRAIN TEST SPLIT  """

input_names = ["create_date"]
output_names = ["targ_leads"]

full_data = nn_data[["create_date", "targ_leads"]]

full_data["create_date"] = full_data["create_date"].apply(time_2ms)

scaler = MinMaxScaler()
full_data=pd.DataFrame(scaler.fit_transform(full_data),
                       columns=full_data.columns,
                       index=full_data.index)
full_data['type_channel'] = nn_data['type_channel']


"""  TRAIN MODEL AND SAVE """

for el in full_data['type_channel'].unique():
  sort_data = full_data[full_data['type_channel'] == el]

  train_data = sort_data[input_names + output_names]

  train = np.asarray(train_data[input_names]).astype('float32')
  target = np.asarray(train_data[output_names]).astype('float32')

  X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=0)

  model = keras.Sequential()

  model.add(keras.layers.Dense(units=64, activation='relu'))
  model.add(keras.layers.Dense(units=128, activation='relu'))
  model.add(keras.layers.Dense(units=256, activation='relu'))
  model.add(keras.layers.Dense(units=1, activation='relu'))

  opt = tf.keras.optimizers.Adam(learning_rate=0.001)

  model.compile(loss='mean_squared_error', optimizer=opt)

  fit_results = model.fit(x=X_train, y=y_train, epochs=20, validation_split=0.2, batch_size=32)

  predicted_test = model.predict(X_val)
  y_pred = predicted_test

  print(el)
  print(mean_squared_error(y_val, y_pred))

  model.save(f"{el}.h5")