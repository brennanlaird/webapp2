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


    price_fig = go.Figure()

    price_fig.add_trace(go.Scatter(x=list(price_data.index), y=list(price_data['Close'])))

    title = ticker + " Price"
    price_fig.update_layout(title_text=title)

    priceJSON = json.dumps(price_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # print(price_data.head())
    # print(price_data.info())
    return render_template("index.html", ticker=ticker, stock_list=stock_list, priceJSON=priceJSON)


if __name__ == '__main__':
    app.run()
