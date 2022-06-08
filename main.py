from turtle import color
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
plt.figure(figsize=(20,10))
sns.lineplot(x = "Date", y = "Close" , data = sensex_data)
plt.title("Closing price of SENSEX from 2001 to 2022", size = 15, color = "blue")
plt.xticks(size = "15")
plt.yticks(size = "15")
plt.xlabel("Time in Years", size = 15)
plt.ylabel("Closing Price", size = 15)
plt.show()