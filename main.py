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




