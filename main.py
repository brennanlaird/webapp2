from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
import json
import plotly
import plotly.graph_objects as go
import pandas_ta as ta
import warnings
from statsmodels.tsa.arima.model import ARIMA

from tqdm import tqdm

warnings.filterwarnings("ignore")

app = Flask(__name__)
stock_list = ['AAPL', 'GME', 'MSFT', 'TSLA']

# Initialize an empy data frame outside functions
price_data = pd.DataFrame()


# @ is a decorator - a way to wrap a function and modify its behavior
@app.route('/', methods=['GET', 'POST'])
def index():
    stock_list = ['AAPL', 'GME', 'MSFT', 'TSLA']

    if request.method == 'POST':
        ticker = request.form['stock']
    return render_template("index.html", stock_list=stock_list)


@app.route('/get_data', methods=['GET', 'POST'])
def get_data():
    ticker = request.form['stock']
    global price_data
    stock_obj = yf.Ticker(ticker)
    price_data = stock_obj.history(period='1y')
    price_data = price_data.drop('Dividends', 1)
    price_data = price_data.drop('Stock Splits', 1)

    # Adding the 50-day and 200-day moving averages to the dataframe
    price_data['50dayMA'] = price_data.Close.rolling(50).mean()
    price_data['200dayMA'] = price_data.Close.rolling(200).mean()

    # Setting up the price graph
    price_fig = go.Figure()

    price_fig.add_trace(go.Scatter(x=list(price_data.index), y=list(price_data['Close'])))

    title = ticker + " Price"
    price_fig.update_layout(title_text=title)

    # Add range slider and time step buttons to the price figure
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

    # Setting up the volume graph
    vol_fig = go.Figure()

    vol_fig.add_trace(go.Scatter(x=list(price_data.index), y=list(price_data['Volume'])))

    title = ticker + " Volume"
    vol_fig.update_layout(title_text=title)

    # Add range slider to the volume figure
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

    # Setting up the moving average figure
    ma_fig = go.Figure()

    ma_fig.add_trace(go.Scatter(x=list(price_data.index), y=list(price_data['Close'])))
    ma_fig.add_trace(go.Scatter(x=list(price_data.index), y=list(price_data['50dayMA'])))
    ma_fig.add_trace(go.Scatter(x=list(price_data.index), y=list(price_data['200dayMA'])))

    title = ticker + " Moving Averages"
    ma_fig.update_layout(title_text=title)

    # Add range slider and buttons to the moving average figure
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

    # Convert figures to javascript to display on webpage
    priceJSON = json.dumps(price_fig, cls=plotly.utils.PlotlyJSONEncoder)
    volJSON = json.dumps(vol_fig, cls=plotly.utils.PlotlyJSONEncoder)
    maJSON = json.dumps(ma_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # print(price_data.head())
    # print(price_data.info())
    return render_template("index.html", ticker=ticker, stock_list=stock_list, priceJSON=priceJSON, volJSON=volJSON,
                           maJSON=maJSON)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    ml_text = "Model results display here"
    global price_data
    # Adds the exponential moving average to the data set and then truncates
    price_data.ta.ema(close='Close', length=10, append=True)
    price_data_trunc = price_data.iloc[10:]

    predictions = []
    history = price_data_trunc[0:len(price_data_trunc)]

    # Puts the EMA prices into a list
    history = [i for i in price_data_trunc['EMA_10']]

    # Puts the history into a dataframe
    history_df = pd.DataFrame(history)

    # Create a 30 day rolling forecat
    for i in tqdm(range(30)):
        # Runs and fits the ARIMA model with the specified order
        model = ARIMA(history, order=(2, 1, 3))
        model_fit = model.fit()

        # Uses the forecast method to predict a single future timestep
        next_forecast = model_fit.forecast()

        # Gets the value of the next prediction
        next_pred = next_forecast[0]

        # pd.concat(history,next_forecast)
        history.append(next_pred)
        predictions.append(next_pred)

        print("Loop ", i, " prediction : ", next_pred)

    # print(price_data.head())
    # print(price_data.info())
    #
    # print(price_data_trunc.head())
    #print(price_data_trunc.info())

    pred_df = pd.DataFrame(predictions)
    pred_df.index = pred_df.index + len(history) - 30

    # Setting up the moving average figure
    pred_fig = go.Figure()

    pred_fig.add_trace(go.Scatter(x=list(history_df.index), y=list(history_df[0])))
    pred_fig.add_trace(go.Scatter(x=list(pred_df.index), y=list(pred_df[0])))

    title = "30-Day Price Forecast"
    pred_fig.update_layout(title_text=title)
    predJSON = json.dumps(pred_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("index.html", ml_text=ml_text, predJSON=predJSON)


if __name__ == '__main__':
    app.run()


def store_data():
    return ()
