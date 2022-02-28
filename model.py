# import statements for Python code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import math
import pandas_ta
import itertools
from tqdm.notebook import tqdm
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

import warnings

warnings.filterwarnings("ignore")
# Additional import statements specific to the ML portion
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg

from statsmodels.graphics.tsaplots import plot_acf
# from pmdarima.arima import auto_arima
# imports for use in creating the web app portion
import ipywidgets as widgets
from ipywidgets import interactive, interact
from IPython.display import display, clear_output


def get_data(ticker):
    # Import the CSV file
    # This should be converted to a function that will be called based on the ticker symbol passed via the Anvil code.

    path = 'C:\\Users\\brenn\\Documents\\School\\C964 - Capstone\\Datasets\\Stocks and ETF\\Stocks\\'

    extension = '.us.txt'
    file_name = path + ticker + extension

    # Read the specified price into a dataframe
    price_data = pd.read_csv(file_name, index_col='Date', parse_dates=['Date'])

    price_fig = go.Figure()

    price_fig.add_trace(go.Scatter(x=list(price_data.index), y=list(price_data['Close'])))

    title = ticker + " Price"
    price_fig.update_layout(title_text=title)

    # Add range slider
    price_fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    price_fig.show()

    vol_fig = go.Figure()

    vol_fig.add_trace(go.Scatter(x=list(price_data.index), y=list(price_data['Volume'])))

    title = ticker + " Volume"
    vol_fig.update_layout(title_text=title)

    # Add range slider
    vol_fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    vol_fig.show()

    # Adding the 50-day and 200-day moving averages to the dataframe
    price_data['50dayMA'] = price_data.Close.rolling(50).mean()
    price_data['200dayMA'] = price_data.Close.rolling(200).mean()

    ma_fig = go.Figure()

    ma_fig.add_trace(go.Scatter(x=list(price_data.index), y=list(price_data['Close'])))
    ma_fig.add_trace(go.Scatter(x=list(price_data.index), y=list(price_data['50dayMA'])))
    ma_fig.add_trace(go.Scatter(x=list(price_data.index), y=list(price_data['200dayMA'])))

    title = ticker + " Moving Averages"
    ma_fig.update_layout(title_text=title)

    # Add range slider
    ma_fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    ma_fig.show()

    price_data.ta.ema(close='Close', length=10, append=True)

    price_data_trunc = price_data.iloc[10:]


ticker_box = widgets.Dropdown(options=['', 'AAPL', 'GME'])
ticker_box.value = ''
display(ticker_box)


def on_data_click(b):
    with out:
        clear_output()
        ticker = ticker_box.value
        if ticker == '':
            # Do nothing
            print("Nothing selected")
            return
        else:
            get_data(ticker)
            return ticker


data_button = widgets.Button(description='Get Data', disabled=False)
out = widgets.Output()
# ticker = ticker_box.value


# linking button and function together using a button's method
data_button.on_click(on_data_click)

widgets.VBox([data_button, out])
