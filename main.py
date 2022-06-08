import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM

sensex_data = yf.download("^BSESN", start = "2001-01-01", end = "2022-06-01")

sensex_data.head()

print(sensex_data)

