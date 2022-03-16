from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import yfinance as yf
import json
import plotly
import plotly.graph_objects as go
import pandas_ta as ta
import warnings
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

# Filters out warnings related to future versions of the pandas library
warnings.filterwarnings("ignore")

# Sets up a Flask object
app = Flask(__name__)

# Sets the secret key for sessions. THIS SHOULD BE KEPT SECRET!
app.secret_key = 'thesecretkeyissecret'

# Set the list of supported stocks
stock_list = ['AAPL', 'GME', 'MSFT', 'TSLA']


def make_prediction(price_data_trunc):
    # Set a blank predictions list to store the predictions made
    predictions = []

    # Sets up a history list to store the price history and the predictions together
    history = price_data_trunc[0:len(price_data_trunc)]

    # Puts the EMA prices into a list
    history = [i for i in price_data_trunc['EMA_10']]

    # Puts the history into a dataframe
    history_df = pd.DataFrame(history)

    # Variable to control the forecast length
    f_len = 30

    # Create a rolling forecast based on the f_len variable
    for i in tqdm(range(f_len)):
        # Runs and fits the ARIMA model with the specified order
        model = ARIMA(history, order=(2, 0, 1))
        model_fit = model.fit()

        # Uses the forecast method to predict a single future time step
        next_forecast = model_fit.forecast()

        # Gets the value of the next prediction
        next_pred = next_forecast[0]

        # Appends the forecast to the history and predictions list objects
        history.append(next_pred)
        predictions.append(next_pred)

    # Sets up the min and max values for the y-axis of the prediction graph
    min_y = min(history[len(history) - 60: len(history)]) * 0.95
    max_y = max(history[len(history) - 60: len(history)]) * 1.05

    # Creates a data frame object from the predictions list
    pred_df = pd.DataFrame(predictions)

    # Indexes the prediction data frame, so it can be graphed on the same scale as the history.
    pred_df.index = pred_df.index + len(history) - f_len

    # Creates the graph of the predictions and history on the same plat
    pred_fig = go.Figure()

    pred_fig.add_trace(go.Scatter(x=list(history_df.index), y=list(history_df[0]), name="Price"))
    pred_fig.add_trace(go.Scatter(x=list(pred_df.index), y=list(pred_df[0]), name="Prediction"))
    pred_fig.update_xaxes(range=[len(history) - 60, len(history)])
    pred_fig.update_yaxes(range=[min_y, max_y])
    title = "30-Day Price Forecast"
    pred_fig.update_layout(title_text=title)

    # Creates the javascript object of the graph and returns it to the calling function
    predJSON = json.dumps(pred_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return predJSON


# The login function
@app.route('/login', methods=['GET', 'POST'])
def login():
    # Initialize the login error to None
    login_error = None

    # runs the login procedures if the post method is used
    if request.method == 'POST':
        # Sets the username based on the form
        username = request.form.get('username')

        # Detects if the username and password match the values required
        if request.form['username'] != 'dfadmin' or request.form['password'] != 'dfadmin':
            # If the values do not match, change the login error and retunr the login page to display the error
            login_error = "Login Error. Try again."
            return render_template('login.html', login_error=login_error)
        else:
            # If login is successful set the username as the active name for the session
            session['user'] = username
            # redirects to the main page on successful login
            return redirect(url_for('index'))
    return render_template('login.html', login_error=login_error)


# @ is a decorator - a way to wrap a function and modify its behavior
@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    # If the user has successfully logged in then proceed to the main page
    if 'user' in session:
        # Sets the prediction button not to display
        display_status = 'hidden'
        return render_template("index.html", stock_list=stock_list, display_status=display_status)
    # If the user is not in the session then the user has not logged in and is redirected to the login page
    return redirect(url_for('login'))

# Function that runs when the get data button is pushed.
@app.route('/get_data', methods=['GET', 'POST'])
def get_data():
    # Gets the ticker symbol from the web form
    ticker = request.form['stock']

    # Sets up an empty data frame to store price data
    price_data = pd.DataFrame()

    # Runs the get prices function with the selected ticker to populates the price data DF
    price_data = get_prices(ticker, 'max')

    # Sets the ticker selected to a session variable. This allows the value to persist between functions
    session['ticker'] = ticker
    # Adding the 50-day and 200-day moving averages to the dataframe
    price_data['50dayMA'] = price_data.Close.rolling(50).mean()
    price_data['200dayMA'] = price_data.Close.rolling(200).mean()

    # Setting up the price graph
    price_fig = go.Figure()

    # Sets the column from the DF to graph
    price_fig.add_trace(go.Scatter(x=list(price_data.index), y=list(price_data['Close'])))

    # Sets the title of the graph
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

    # Sets the trace from the DF to add to the graph
    vol_fig.add_trace(go.Scatter(x=list(price_data.index), y=list(price_data['Volume'])))

    # Sets the title of the volume graph
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

    # Sets the traces from the DF to add to the graph and adds their names for the legend
    ma_fig.add_trace(go.Scatter(x=list(price_data.index), y=list(price_data['Close']), name="Closing Price"))
    ma_fig.add_trace(go.Scatter(x=list(price_data.index), y=list(price_data['50dayMA']), name="50-Day MA"))
    ma_fig.add_trace(go.Scatter(x=list(price_data.index), y=list(price_data['200dayMA']), name="200-Day MA"))

    # Adds the title for the moving average graphs
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

    # Set the prediction button to be visible
    display_status = 'visible'
    return render_template("index.html", ticker=ticker, stock_list=stock_list, priceJSON=priceJSON, volJSON=volJSON,
                           maJSON=maJSON, display_status=display_status)


# Function that runs whenever the Run prediction button is pressed
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Get the value of the session ticker
    ticker = session['ticker']

    # Sets the text to display on the webpage
    ml_text = ticker + " 30-Day Price Forecast"

    # Sets up an empty data frame to store price data
    price_data = pd.DataFrame()

    # Runs the get prices function with the selected ticker to populates the price data DF
    price_data = get_prices(ticker, '1y')

    # Adds the exponential moving average to the data set and then truncates to remove NULL values
    price_data.ta.ema(length=10, append=True)
    price_data_trunc = pd.DataFrame(price_data.iloc[10:])

    # Uses the make prediction function by passing the truncated price data. A javascript graph object is returned
    predJSON = make_prediction(price_data_trunc)

    # Set the prediction button to be hidden
    display_status = 'hidden'

    return render_template("index.html", ml_text=ml_text, predJSON=predJSON, ticker=ticker, stock_list=stock_list,
                           display_status=display_status)


# Function to retrieve stock price info using the yfinance library
def get_prices(ticker, time):
    # Creates a stock object by retrieving information based on the ticker passed into the function
    stock_obj = yf.Ticker(ticker)

    # Sets up a blank data frame to store price data
    prices = pd.DataFrame

    # Puts the price history for the defined period into the prices DF
    prices = stock_obj.history(period=time)

    # Drop the Dividends and Stock Splits columns as they are not used
    prices = prices.drop('Dividends', 1)
    prices = prices.drop('Stock Splits', 1)

    # Return the prices data frame
    return prices


# Runs the main Flask app
if __name__ == '__main__':
    app.run()
