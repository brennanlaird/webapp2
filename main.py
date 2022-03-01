from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
import json
import plotly
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")



app = Flask(__name__)
stock_list = ['AAPL', 'GME', 'MSFT', 'TSLA']


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
    return render_template("index.html", ticker=ticker, stock_list=stock_list, priceJSON=priceJSON, volJSON=volJSON, maJSON=maJSON)


if __name__ == '__main__':
    app.run()
