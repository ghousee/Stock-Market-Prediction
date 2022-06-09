from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#Loading Data

start = dt.datetime(2001,1,1)
end = dt.datetime(2022,1,1)

sensex_data = web.DataReader("^BSESN",'yahoo',start ,end) ##Webscraping

#print(sensex_data)
#Data Cleaning



sensex_data = sensex_data.drop_duplicates()

#Checking Null Values
#print(sensex_data.isnull().sum())

#Statistical analysis
print(sensex_data.describe())

print(sensex_data)

#Closing Price Graph
plt.figure(figsize=(10,7))
sns.lineplot(x = "Date", y = "Close" , data = sensex_data)
plt.title("Closing price of SENSEX from 2001 to 2022", size = 15, color = "blue")
plt.xticks(size = "15")
plt.yticks(size = "15")
plt.xlabel("Time in Years", size = 15)
plt.ylabel("Closing Price", size = 15)

plt.show()

#Simple Moving Average(sma)

#def movingaverage(values,window):
#    weights = np.repeat(1.0,window) / window
 #   smas = np.convolve(values,weights,'valid')
  #  return smas

#Plotting 30-day moving average(Rolling Mean) & Standard deviation
plt.figure(figsize=(10,7))
sns.lineplot(x = "Date", y = "Close" , data = sensex_data)
rollng_mean = sensex_data.rolling(window = 30).mean()['Close']
sns.lineplot(x = "Date", y = rollng_mean, data = sensex_data, label = "Moving Average")
rollng_std = sensex_data.rolling(window = 30).std()["Close"]
sns.lineplot(x = "Date", y = rollng_std, data = sensex_data, label = "Rolling Standard Deviation")
plt.title("Closing price, 30 Simple Moving Average & Standard Deviation of SENSEX from 2001 to 2022", size = 15, color = "blue")
plt.xticks(size = "15")
plt.yticks(size = "15")
plt.xlabel("Time in Years", size = 15)
plt.ylabel("Closing Price", size = 15)
plt.show()


#Plotting Profit/Loss in percentage
PL = sensex_data["Close"] / sensex_data["Close"].shift(1) - 1
plt.figure(figsize=(10,7))
PL.plot(label = 'P/L', color = 'green')
plt.title("Profit/Loss")
plt.show()

#Modeling the data

xaxis = sensex_data[["Open","High","Low","Close"]]
yaxis = sensex_data["Adj Close"]

#Using LSTM(Long Short-Term Memory) after rescaling,Feature Range of (-1,1)
##dividing dataset into Training and Test

xtrain, xtest, ytrain, ytest = train_test_split(xaxis, yaxis, test_size = 0.3, random_state = 3)

#Prediction
linreg = LinearRegression()
linreg.fit(xtrain, ytrain)
predic = linreg.predict(xtest)
print(predic[:6])

#Using pandas, create a df
DF = pd.DataFrame({"Actual: ": ytest, "Predicted: ": predic}).tail()
print(DF)

#Plotting actual and predicted values
plt.figure(figsize = (10,5))
sns.kdeplot(data = xtest, x = ytest, label = "Actual Values")
sns.kdeplot(data = xtest, x = predic, label = "Predicted Values")
plt.legend()
plt.title("Actual Values v/s Predicted Values", size = 15, color = "red")
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.xlabel("Adjusted Close", size = 15, color = "blue")
plt.ylabel("Density", size = 15, color = "red")
plt.show()
plt.show()

#Model Evaluation
me = metrics.mean_squared_error(ytest, predic)
print(me)