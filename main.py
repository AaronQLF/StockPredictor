import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

with st.sidebar:
    st.title('Stock Price Prediction')
    st.write('This is a simple web app to predict the price of a stock using a machine learning model.')
    st.write('Please adjust the values of the input features to get the prediction.')
    st.write('This app is for educational purposes only.')
    st.write('Please do not use the predictions for trading purposes.')

# Select the ticker
ticker = st.sidebar.text_input('Enter the stock ticker (e.g. AAPL):')

# Select how far in the future we want to predict
days_to_predict = st.sidebar.slider('Days to predict:', 1, 60, 5)

# Select the model
model_name = st.sidebar.selectbox('Select the model:', ['Linear Regression', 'Random Forest', "LSTM"])

# Select the number of lags
n_lags = st.sidebar.slider('Number of lag days (for models):', 1, 60, 5)

# Select features
features_selected = st.sidebar.multiselect('Select the features:', ['Open', 'High', 'Low', 'Close', 'Volume'], ['Close'])

# Select the split size
split_size = st.sidebar.slider('Training data proportion:', 0.1, 0.9, 0.8)

# Select the number of estimators
n_estimators = st.sidebar.slider('Number of estimators (for Random Forest):', 1, 1000, 100)

# Select the max depth
max_depth = st.sidebar.slider('Max depth (for Random Forest):', 1, 100, 10)

# Select the learning rate
learning_rate = st.sidebar.slider('Learning rate (for LSTM):', 0.001, 0.1, 0.01, step=0.001, format="%.3f")

# Select the number of epochs
epochs = st.sidebar.slider('Number of epochs (for LSTM):', 10000, 100000, 10000)

# Select the batch size
batch_size = st.sidebar.slider('Batch size (for LSTM):', 1, 256, 32)

# Function to load data from yfinance
def load_data(ticker, period='5y', interval='1d'):
    if ticker:
        try:
            data = yf.download(ticker, period=period, interval=interval)
            if data.empty:
                st.warning('No data found for the ticker. Please enter a valid stock ticker.')
                return None
            return data
        except Exception as e:
            st.warning('Error loading data: {}'.format(e))
            return None
    else:
        return None

# Load data
data = load_data(ticker)

if data is not None:
    # Prepare the data based on the selected features
    if model_name in ['Linear Regression', 'Random Forest']:
        # Create lagged features based on selected features
        for feature in features_selected:
            for i in range(1, n_lags + 1):
                data[f'{feature}_lag_{i}'] = data[feature].shift(i)

        # Create the target variable
        data['Target'] = data['Close'].shift(-days_to_predict)

        data = data.dropna()

        # Define the features and target
        lagged_features = [f'{feature}_lag_{i}' for feature in features_selected for i in range(1, n_lags + 1)]
        target = 'Target'

        # Check if enough data is available
        if data.shape[0] < 1:
            st.error("Not enough data available after processing. Please reduce 'Number of lag days' or 'Days to predict'.")
        else:
            # Split the data
            split_index = int(len(data) * split_size)
            train_data = data.iloc[:split_index]
            test_data = data.iloc[split_index:]

            # Create the features and the target variable
            X_train = train_data[lagged_features].values
            y_train = train_data[target].values
            X_test = test_data[lagged_features].values
            y_test = test_data[target].values

            if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                st.error("Training or testing data is empty after splitting. Please adjust the 'Training data proportion'.")
            else:
                if model_name == 'Linear Regression':
                    model_lr = LinearRegression()
                    model_lr.fit(X_train, y_train)

                    # Make predictions on the test set
                    y_pred = model_lr.predict(X_test)

                    # Plot the predictions vs actual prices
                    st.subheader('Stock Price Predictions vs Actual')
                    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=test_data.index)
                    st.line_chart(predictions_df)

                    # Predict future stock prices
                    st.subheader('Future Stock Price Predictions')

                    last_known_features = data[lagged_features].iloc[-1].values.reshape(1, -1)
                    future_predictions = []

                    for _ in range(days_to_predict):
                        next_prediction = model_lr.predict(last_known_features)
                        future_predictions.append(next_prediction[0])
                        # Update last_known_features with the new prediction
                        last_known_features = np.roll(last_known_features, -1)
                        last_known_features[0, -1] = next_prediction[0]

                    future_dates = pd.date_range(start=data.index[-2] + pd.Timedelta(days=1), periods=days_to_predict, freq='B')
                    future_predictions_df = pd.DataFrame({'Future Predictions': future_predictions}, index=future_dates)

                    # Combine the actual and future predictions for plotting
                    combined_df = pd.concat([data['Close'], future_predictions_df['Future Predictions']], axis=0)
                    st.line_chart(combined_df)

                elif model_name == 'Random Forest':
                    model_rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                    model_rf.fit(X_train, y_train)

                    # Make predictions on the test set
                    y_pred_test = model_rf.predict(X_test)

                    # Create a DataFrame for the test predictions
                    test_predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test}, index=test_data.index)

                    # Future predictions
                    st.subheader('Future Stock Price Predictions')

                    last_known_features = data[lagged_features].iloc[-1].values.reshape(1, -1)
                    future_predictions = []

                    for _ in range(days_to_predict):
                        next_prediction = model_rf.predict(last_known_features)
                        future_predictions.append(next_prediction[0])
                        # Update last_known_features with the new prediction
                        last_known_features = np.roll(last_known_features, -1)
                        last_known_features[0, -1] = next_prediction[0]

                    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)
                    future_predictions_df = pd.DataFrame({'Predicted': future_predictions}, index=future_dates)

                    # Combine the actual 'Close' prices, test predictions, and future predictions
                    combined_df = pd.concat([data['Close'], future_predictions_df['Predicted']], axis=0)
                    st.subheader('Combined Actual and Predicted Stock Prices')
                    st.line_chart(combined_df)

    elif model_name == 'LSTM':
        # Prepare data for LSTM
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[features_selected])

        time_steps = n_lags  # Using n_lags as time steps

        # Check if enough data is available
        if len(scaled_data) - time_steps - days_to_predict + 1 <= 0:
            st.error("Not enough data to create sequences for the given number of lags and days to predict. Please reduce 'Number of lag days' or 'Days to predict'.")
        else:
            # Create sequences
            def create_sequences(input_data, time_steps):
                X, y = [], []
                for i in range(len(input_data) - time_steps - days_to_predict + 1):
                    X.append(input_data[i:(i + time_steps)])
                    y.append(input_data[i + time_steps + days_to_predict - 1, 0])  # Only select the 'Close' feature for the target variable
                return np.array(X), np.array(y)

            X, y = create_sequences(scaled_data, time_steps)

            # Split into training and testing sets
            split = int(X.shape[0] * split_size)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                st.error("Training or testing data is empty after splitting. Please adjust the 'Training data proportion' or reduce 'Number of lag days' or 'Days to predict'.")
            else:
                # Reshape input data to 3D (samples, time_steps, features)
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], len(features_selected)))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], len(features_selected)))

                # Build LSTM model
                model_lstm = Sequential()
                model_lstm.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
                model_lstm.add(Dense(1))  # Predicting 1 output: the 'Close' price
                model_lstm.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

                # Train the model
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model_lstm.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

                # Make predictions
                y_pred_scaled = model_lstm.predict(X_test)
                y_pred = scaler.inverse_transform(y_pred_scaled)[:, 0]  # Inverse transform the 'Close' column only
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]

                # Calculate the mean squared error
                mse = mean_squared_error(y_test_actual, y_pred)
                st.write('Mean Squared Error on Test Set:', mse)

                # Plot predictions
                predictions_df = pd.DataFrame({'Actual': y_test_actual.flatten(), 'Predicted': y_pred.flatten()}, index=data.index[-len(y_test_actual):])
                st.subheader('Stock Price Predictions vs Actual')
                st.line_chart(predictions_df)

                # Future predictions
                st.subheader('Future Stock Price Predictions')

                # Get the last sequence from data
                last_sequence = scaled_data[-time_steps:].reshape((1, time_steps, len(features_selected)))
                future_predictions = []

                for _ in range(days_to_predict):
                    next_pred_scaled = model_lstm.predict(last_sequence)
                    next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]  # Predicting only the 'Close' price
                    future_predictions.append(next_pred)

                    # Update the last_sequence with the new prediction
                    next_pred_scaled_reshaped = next_pred_scaled.reshape(1, 1, len(features_selected))
                    last_sequence = np.concatenate((last_sequence[:, 1:, :], next_pred_scaled_reshaped), axis=1)

                future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days_to_predict, freq='B')
                future_predictions_df = pd.DataFrame({'Future Predictions': future_predictions}, index=future_dates)

                # Combine the actual and future predictions for plotting
                combined_df = pd.concat([data['Close'], future_predictions_df['Future Predictions']], axis=0)
                st.line_chart(combined_df)

else:
    if not ticker:
        st.warning("Please enter a stock ticker.")
    else:
        st.warning("Data not available or insufficient data.")
