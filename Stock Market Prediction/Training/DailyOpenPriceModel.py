# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 19:22:48 2021

@author: udaia
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from glob import glob
import os

CWD = os.getcwd()

files = glob(CWD + "/Data/*.csv")


df = pd.DataFrame()
for i in files:
    temp = pd.read_csv(i)
    df = df.append(temp)
df.shape

companies = df["Symbol"].unique()
companies[:5]

# df = df[df["Symbol"] == companies[0]]
df = df[df["Symbol"] == 'RELIANCE']

c = 'RELIANCE'

df["Date"] = pd.to_datetime(df["Date"])
df["Date"][120:].shape

df = df.sort_values(by = ["Date"], ascending=True)

training_set = df.loc[df["Symbol"] == c, "Open"].values
print(f"training on: {c}")
print(f"training data points: {len(training_set)}")
training_set = training_set.reshape(-1,1)
training_set[:5]

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler (feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled

x_train = []
y_train = []
# 60 timesteps and 1 output
for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = x_train[..., np.newaxis]
y_train = y_train[..., np.newaxis]

print(x_train.shape)
print(y_train.shape)

## Modeling
> training on 60 days timestep

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

# Adding the first LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences=True, 
                   input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some dropout regularisation
regressor.add(LSTM(units=50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output Layer
regressor.add(Dense(units=1))

regressor.summary()

regressor.compile(optimizer="adam", loss="mean_squared_error")

regressor.fit(x_train, y_train, epochs=10, batch_size=32)

## Prediction test on same company data

dataset_total = df[df["Symbol"] == c]["Open"]
inputs = dataset_total.values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
x_test = []

for i in range(60, len(inputs)):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_test.shape

y_pred = regressor.predict(x_test)
y_pred = sc.inverse_transform(y_pred)
y_pred.shape

plt.figure(figsize=(12, 5))
plt.plot(df[df["Symbol"] == c]["Date"][60:], y_pred.flatten(), label="Predictions")
plt.plot(df[df["Symbol"] == c]["Date"], df[df["Symbol"] == c]["Open"], label="Actual", alpha=0.4)
plt.legend(loc=(1.01, 0.85))
plt.show()

check = pd.DataFrame(np.column_stack([df[df["Symbol"] == c]["Date"].values[60:].astype(str), 
                              df[df["Symbol"] == c]["Open"].values[60:], y_pred]),
             columns = ["Date", "True", "Predicted"]
            )
check["Date"] = pd.to_datetime(check["Date"])
check[["True", "Predicted"]] = check[["True", "Predicted"]].astype(float)

plt.figure(figsize=(15, 5))
plt.plot(check[check["Date"] > ("2011-01-01")]["True"])
plt.plot(check[check["Date"] > ("2011-01-01")]["Predicted"])


check = check.sort_values(by=["Date"], ascending=True)

# Rolling Match
rolling = 60
check["Previous"] = check["True"].shift(1)
check["Rolling_mean_predicted"] = check.rolling(rolling)["Predicted"].mean()
check["Rolling_mean_true"] = check.rolling(rolling)["True"].mean()
# check["Rolling_5_max"] = check.rolling(5)["Predicted"].max()

plt.figure(figsize=(15, 5))
plt.plot(check[check["Date"] > ("2011-01-01")]["Rolling_mean_predicted"], label="Predicted")
plt.plot(check[check["Date"] > ("2011-01-01")]["Rolling_mean_true"], label="True")
plt.legend(loc=(1, 0.85))
plt.show()

check["Trend_True"] = ""
check.loc[check["Rolling_mean_true"] > check["Previous"], "Trend_True"] = "Increasing"
check.loc[check["Rolling_mean_true"] == check["Previous"], "Trend_True"] = "Stable"
check.loc[check["Rolling_mean_true"] < check["Previous"], "Trend_True"] = "Decreasing"


check["Trend_Predicted"] = ""
check.loc[check["Rolling_mean_predicted"] > check["Previous"], "Trend_Predicted"] = "Increasing"
check.loc[check["Rolling_mean_predicted"] == check["Previous"], "Trend_Predicted"] = "Stable"
check.loc[check["Rolling_mean_predicted"] < check["Previous"], "Trend_Predicted"] = "Decreasing"

# check.head()

check["Trend_Match"] = ""
check.loc[check["Trend_True"] == check["Trend_Predicted"], "Trend_Match"] = 1
check.loc[check["Trend_True"] != check["Trend_Predicted"], "Trend_Match"] = 0

print(f"Percentage Match: { round(check['Trend_Match'].sum() * 100 / len(check), 2)}")

## Next Steps
> Make predictions on predictions for the next 5 days (5 iterations).
* Checking Accuracy based on 
    * Ground Truth and 
    * Predictions on Ground Truth