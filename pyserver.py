import base64
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from lstm import machinelearningcode
import os
import concurrent.futures
import pandas as pd
from calculate_optimal_weights import calculate_optimal_weights
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

app = Flask(__name__)
CORS(app)

# Modify the run() function to return the data as a dictionary instead of a list


def run(stock_name):
    predictions, test_rmse = machinelearningcode(stock_name)
    return predictions, test_rmse


@app.route('/predict', methods=['POST'])
def handle_predict():
    data = request.json
    # Split the comma-separated string into a list of stock names
    stock_names = data.get('prediction').split(",")
    print("Stock names received:", stock_names)

    predictions = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each stock name to the executor
        future_to_stock = {executor.submit(
            run, stock_name): stock_name for stock_name in stock_names}

        # Retrieve the results as they become available
        for future in concurrent.futures.as_completed(future_to_stock):
            stock_name = future_to_stock[future]
            try:
                stock_prediction, test_rmse = future.result()
                predictions[stock_name] = stock_prediction
            except Exception as e:
                print(
                    f"An error occurred while processing {stock_name}: {str(e)}")

    return jsonify(predictions)


@app.route('/optimize-portfolio', methods=['POST'])
def optimize_portfolio():
    data = request.json
    stock_names = data.get('stockNames').split(",")
    stock_weights = data.get('stockWeights')
    stock_predictions = data.get('stockPrediction')

    # Calculate the optimal weights based on the stock data and predictions
    optimal_weights = calculate_optimal_weights(stock_names, stock_predictions)

    # Filter the optimal weights based on the provided stock weights
    filtered_weights = {
        stock: optimal_weights[stock] for stock in stock_weights.keys()}

    return jsonify(filtered_weights)


if __name__ == '__main__':
    app.run(port=5000, threaded=True)
