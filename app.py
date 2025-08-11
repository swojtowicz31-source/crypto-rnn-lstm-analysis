import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense

# Parametry danych
symbol = 'BTC-USD'
start_date = '2020-01-01'
end_date = '2023-01-01'
epochs = 100
batch_size = 16  # Stała wartość batch size
timesteps_list = [20, 50, 100]
model_types = ['LSTM', 'GRU', 'SimpleRNN']

# Pobranie danych giełdowych
data = yf.download(symbol, start=start_date, end=end_date)
close_prices = data['Close'].values

# Normalizacja danych
close_prices = (close_prices - np.mean(close_prices)) / np.std(close_prices)
close_prices = (close_prices - np.min(close_prices)) / (np.max(close_prices) - np.min(close_prices))

# Funkcja do przygotowania próbek
def create_dataset(data, timesteps):
    X, Y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i + timesteps])
        Y.append(data[i + timesteps])
    return np.array(X), np.array(Y)

# Funkcja tworząca model
def create_model(model_type, timesteps):
    if model_type == 'LSTM':
        return Sequential([
            LSTM(100, activation='tanh', input_shape=(timesteps, 1), return_sequences=True),
            LSTM(100, activation='tanh', return_sequences=True),
            LSTM(50, activation='tanh'),
            Dense(1)
        ])
    elif model_type == 'GRU':
        return Sequential([
            GRU(100, activation='tanh', input_shape=(timesteps, 1), return_sequences=True),
            GRU(100, activation='tanh', return_sequences=True),
            GRU(50, activation='tanh'),
            Dense(1)
        ])
    elif model_type == 'SimpleRNN':
        return Sequential([
            SimpleRNN(100, activation='tanh', input_shape=(timesteps, 1), return_sequences=True),
            SimpleRNN(100, activation='tanh', return_sequences=True),
            SimpleRNN(50, activation='tanh'),
            Dense(1)
        ])

# Lista do przechowywania błędów
errors = []

# Główna pętla testująca różne modele i time steps
for model_type in model_types:
    for timesteps in timesteps_list:
        X, Y = create_dataset(close_prices, timesteps)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        train_size = int(0.7 * len(X))
        val_size = int(0.2 * len(X))
        
        X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
        Y_train, Y_val, Y_test = Y[:train_size], Y[train_size:train_size + val_size], Y[train_size + val_size:]
        
        model = create_model(model_type, timesteps)
        model.compile(optimizer='adam', loss='mse')
        
        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=0)
        
        # Generowanie prognoz dla 1000 próbek testowych
        predictions = model.predict(X_test[:1000])  # Ograniczenie do 1000 próbek
        abs_diff = np.abs(Y_test[:1000] - predictions.flatten())
        
        # Tworzenie histogramu błędów
        plt.figure(figsize=(10, 5))
        plt.hist(abs_diff, bins=30, color='skyblue', edgecolor='black', density=True)
        plt.title(f'Błędy dla {model_type}, time steps={timesteps}, batch size={batch_size}')
        plt.xlabel('Błąd absolutny')
        plt.ylabel('Gęstość')
        plt.show()
        
        # Zapis wyników błędów
        errors.append((model_type, timesteps, batch_size, np.mean(abs_diff), np.min(abs_diff), np.max(abs_diff)))

        # Wizualizacja procesu trenowania
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Strata treningowa')
        plt.plot(history.history['val_loss'], label='Strata walidacyjna')
        plt.title(f'Training Loss dla {model_type}, time steps={timesteps}, batch size={batch_size}')
        plt.xlabel('Epoki')
        plt.ylabel('Strata')
        plt.legend()
        plt.show()

# Tworzenie DataFrame z wynikami
error_df = pd.DataFrame(errors, columns=['Model Type', 'Time Steps', 'Batch Size', 'Mean Absolute Error', 'Min Error', 'Max Error'])
print(error_df)
