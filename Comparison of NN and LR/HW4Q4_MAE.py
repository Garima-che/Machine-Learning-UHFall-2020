# -*- coding: utf-8 -*-
"""5 features NNR and LR using keras_MAE.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uGVTHajQ9fuapA8oOMfcES9jarjKihf6

# Online News Shares

## Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""## Importing the dataset"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
sns.set_style('whitegrid')
df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/OnlineNewsPopularity.csv')
df.head(5)

"""# Encoding the dataset"""

# No encoding required

"""## Raw data visualisation and statistics"""

df.describe()

# We drop the non-predicting features as mentioned in the dataset description file
df= df.drop(['url', ' timedelta'], axis=1)
# We also do feature selection based on the correlation plot in separate program file
df= df.iloc[:,[24,25,28,39,40,58]]

"""## Splitting the dataframe in train and test sets"""

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df.iloc[:,:], test_size = 0.2, random_state = 0)
print(df_train)

ListAttr = []
lengthOfList = len(df)
for i in df:
    print(i)
    ListAttr.append(i)
print(len(ListAttr))

"""## Scaling the train set features"""

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
df_train_scaled= min_max_scaler.fit_transform(df_train)

df_train_scaled= pd.DataFrame(data= df_train_scaled, columns=ListAttr)
print(df_train_scaled)
df_test_scaled= min_max_scaler.transform(df_test)
df_test_scaled= pd.DataFrame(data= df_test_scaled, columns=ListAttr)
print(df_test_scaled)

"""## Defining independent and dependent variables"""

X_train=df_train.iloc[:, 0:-1]
y_train=df_train.iloc[:,-1]
X_test=df_test.iloc[:,0:-1]
y_test=df_test.iloc[:,-1]
print(X_train)
print(y_train)

"""## Training the model

### Linear regression model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import InputLayer

print(tf.__version__)

"""Normalisation layer"""

normalizer = preprocessing.Normalization()

normalizer.adapt(np.array(X_train))

print(normalizer.mean.numpy())

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 3500])
  plt.xlabel('Epoch')
  plt.ylabel('Error [Shares]')
  plt.legend()
  plt.grid(True)

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.predict(X_train[::])

linear_model.layers[1].kernel

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

# Commented out IPython magic to ensure Python compatibility.
# %%time
# history = linear_model.fit(
#     X_train, y_train, 
#     epochs=100,
#     # suppress logging
#     verbose=0,
#     # Calculate validation results on 20% of the training data
#     validation_split = 0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plot_loss(history)
plt.title('Linear regression Model MAE losses across epochs')

test_results={}
test_results['linear_model'] = linear_model.evaluate(X_test, y_test, verbose=0)

test_predictions = linear_model.predict(X_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [shares]')
plt.ylabel('Predictions [shares]')
lims = [0, 15000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.title('Linear rergession performance using MAE')

pd.DataFrame(test_results, index=['Mean absolute error [shares]']).T

"""### Neural network model"""

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.01))
  return model

NN_model = keras.Sequential([
      normalizer,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

NN_model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

NN_model.summary()

# Commented out IPython magic to ensure Python compatibility.
# %%time
# history = NN_model.fit(
#     X_train, y_train,
#     validation_split=0.2,
#     verbose=0, epochs=100)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plot_loss(history)
plt.title('NN Model MAE losses across epochs')

test_results = {}

test_results['NN_model'] = NN_model.evaluate(X_test, y_test, verbose=0)

pd.DataFrame(test_results, index=['Mean absolute error [shares]']).T

test_predictions = NN_model.predict(X_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [shares]')
plt.ylabel('Predictions [shares]')
lims = [0, 30000]
plt.title('NN performace using MSE')
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)