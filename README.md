# Bitcoin Price Prediction with RNN, LSTM, and GRU

This project uses TensorFlow/Keras to train and compare three types of recurrent neural networks — LSTM, GRU, and SimpleRNN — on historical Bitcoin (BTC-USD) price data downloaded from Yahoo Finance. The goal is to predict future prices and analyze prediction errors for different time step settings.

## Features

- Data download and preprocessing from Yahoo Finance
- Models: LSTM, GRU, SimpleRNN with multiple stacked layers
- Testing different time steps: 20, 50, 100
- Visualization of training loss and prediction error distributions
- Comparison of model performance based on absolute error metrics

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- tensorflow
- yfinance

## Installation

```bash
pip install numpy pandas matplotlib tensorflow yfinance
