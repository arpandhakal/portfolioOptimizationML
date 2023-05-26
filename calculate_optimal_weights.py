import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


def calculate_optimal_weights(stock_names, stock_predictions):
    # Calculate the total prediction value
    total_prediction_value = sum(stock_predictions.values())

    # Calculate the expected returns using the stock predictions
    mu = pd.Series(
        {stock: stock_predictions[stock] / total_prediction_value for stock in stock_names})

    # Load the "Close" column of the stock data for the given stock names
    stock_data = {}
    for stock_name in stock_names:
        filename = f"{stock_name}.csv"
        stock_df = pd.read_csv(filename, usecols=["close"])
        stock_data[stock_name] = stock_df

    # Combine the stock data into a single DataFrame
    df = pd.concat(stock_data.values(), axis=1, join='inner')

    # Handle missing values
    df = df.dropna()

    # Calculate the sample covariance matrix
    S = risk_models.sample_cov(df)

    # Create an instance of EfficientFrontier
    ef = EfficientFrontier(mu, S)

    # Optimize for maximal Sharpe ratio
    weights = ef.max_sharpe()

    # Clean the weights to round and clip near-zeros
    clean_weights = ef.clean_weights()

    # Filter the weights based on the provided stock names
    filtered_weights = {stock: clean_weights[stock] for stock in stock_names}

    return filtered_weights
