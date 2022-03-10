from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
import json
import plotly
import plotly.graph_objects as go
import pandas_ta as ta
import warnings
from statsmodels.tsa.arima.model import ARIMA
from celery import Celery
from tqdm import tqdm

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Set the list of supported stocks
stock_list = ['AAPL', 'GME', 'MSFT', 'TSLA']

# Celery configurations
# from Miguel Grinberg blog
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


@celery.task()
def make_prediction(price_data_trunc):
    # The code to run the ML prediction
    print('The celery function worked')
    print(price_data_trunc.info())

    predictions = []
    history = price_data_trunc[0:len(price_data_trunc)]

    # print(price_data_trunc.head())
    # print(price_data_trunc.info())

    # Puts the EMA prices into a list
    history = [i for i in price_data_trunc['EMA_10']]

    # Puts the history into a dataframe
    history_df = pd.DataFrame(history)

    # Variable to control the forecast length
    f_len = 30

    # Create a 30 day rolling forecat
    for i in tqdm(range(f_len)):
        # Runs and fits the ARIMA model with the specified order

        model = ARIMA(history, order=(2, 0, 3))
        model_fit = model.fit()

        # Uses the forecast method to predict a single future timestep
        next_forecast = model_fit.forecast()

        # Gets the value of the next prediction
        next_pred = next_forecast[0]

        # pd.concat(history,next_forecast)
        history.append(next_pred)
        predictions.append(next_pred)

        # print("Loop ", i, " prediction : ", next_pred)

    # print(price_data.head())
    # print(price_data.info())
    #
    # print(price_data_trunc.head())
    # print(price_data_trunc.info())

    # Sets up the min and max values for the y-axis
    min_y = min(history[len(history) - 60: len(history)]) * 0.95
    max_y = max(history[len(history) - 60: len(history)]) * 1.05
    print(min_y)
    print(max_y)

    pred_df = pd.DataFrame(predictions)
    pred_df.index = pred_df.index + len(history) - f_len
    pred_fig = go.Figure()

    pred_fig.add_trace(go.Scatter(x=list(history_df.index), y=list(history_df[0]), name="Price"))
    pred_fig.add_trace(go.Scatter(x=list(pred_df.index), y=list(pred_df[0]), name="Prediction"))
    pred_fig.update_xaxes(range=[len(history) - 60, len(history)])
    pred_fig.update_yaxes(range=[min_y, max_y])
    title = "30-Day Price Forecast"
    pred_fig.update_layout(title_text=title)
    predJSON = json.dumps(pred_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return predJSON


# Celery function to handle the background work associated with the ML model
# Taken from the celery documentation
# def make_celery(celery_app):
#     celery = Celery(
#         celery_app.import_name,
#         backend=celery_app.config['CELERY_RESULT_BACKEND'],
#         broker=celery_app.config['CELERY_BROKER_URL']
#     )
#     celery.conf.update(celery_app.config)
#
#     class ContextTask(celery.Task):
#         def __call__(self, *args, **kwargs):
#             with celery_app.app_context():
#                 return self.run(*args, **kwargs)
#
#     celery.Task = ContextTask
#     return celery


# Initialize an empy data frame outside functions




# @ is a decorator - a way to wrap a function and modify its behavior
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['stock']
    return render_template("index.html", stock_list=stock_list)


@app.route('/get_data', methods=['GET', 'POST'])
def get_data():
    ticker = request.form['stock']
    price_data = pd.DataFrame()
    price_data = get_prices(ticker)

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



    print('Get data finished. Price Data info is next')
    print(price_data.info())
    return render_template("index.html", ticker=ticker, stock_list=stock_list, priceJSON=priceJSON, volJSON=volJSON,
                           maJSON=maJSON)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    ml_text = "Model results display here"

    ticker = "TSLA"
    price_data = pd.DataFrame()
    price_data = get_prices(ticker)



    # Adds the exponential moving average to the data set and then truncates
    price_data.ta.ema(length=10, append=True)

    # print(price_data.head())
    # print(price_data.info())

    price_data_trunc = pd.DataFrame(price_data.iloc[10:])

    print('Price Data Head')
    print(price_data.head())
    print('Price Data Truncated Head')
    print(price_data_trunc.head())
    print('Price Data Info')
    print(price_data.info())
    print('Price Data Truncated Info')
    print(price_data_trunc.info())

    predJSON = make_prediction(price_data_trunc)

    # predictions = []
    # history = price_data_trunc[0:len(price_data_trunc)]
    #
    # #print(price_data_trunc.head())
    # #print(price_data_trunc.info())
    #
    # # Puts the EMA prices into a list
    # history = [i for i in price_data_trunc['EMA_10']]
    #
    # # Puts the history into a dataframe
    # history_df = pd.DataFrame(history)
    #
    # # Variable to control the forecast length
    # f_len = 5
    #
    # # Create a 30 day rolling forecat
    # for i in tqdm(range(f_len)):
    #     # Runs and fits the ARIMA model with the specified order
    #
    #     model = ARIMA(history, order=(2, 0, 3))
    #     model_fit = model.fit()
    #
    #     # Uses the forecast method to predict a single future timestep
    #     next_forecast = model_fit.forecast()
    #
    #     # Gets the value of the next prediction
    #     next_pred = next_forecast[0]
    #
    #     # pd.concat(history,next_forecast)
    #     history.append(next_pred)
    #     predictions.append(next_pred)
    #
    #     # print("Loop ", i, " prediction : ", next_pred)
    #
    # # print(price_data.head())
    # # print(price_data.info())
    # #
    # # print(price_data_trunc.head())
    # # print(price_data_trunc.info())
    #
    # pred_df = pd.DataFrame(predictions)
    # pred_df.index = pred_df.index + len(history) - f_len

    # Setting up the moving average figure
    # pred_fig = go.Figure()
    #
    # pred_fig.add_trace(go.Scatter(x=list(history_df.index), y=list(history_df[0])))
    # pred_fig.add_trace(go.Scatter(x=list(pred_df.index), y=list(pred_df[0])))
    # pred_fig.update_xaxes(range=[len(history) - 60, len(history)])
    # title = "30-Day Price Forecast"
    # pred_fig.update_layout(title_text=title)
    # predJSON = json.dumps(pred_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("index.html", ml_text=ml_text, predJSON=predJSON)


def get_prices(ticker):
    stock_obj = yf.Ticker(ticker)
    prices = pd.DataFrame
    prices = stock_obj.history(period='1y')
    prices = prices.drop('Dividends', 1)
    prices = prices.drop('Stock Splits', 1)
    return prices


if __name__ == '__main__':
    app.run()
