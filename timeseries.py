import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the time series data
data = pd.read_csv('your_data_file.csv')

# Preprocess the data if needed (e.g., convert to datetime, set as index)

# Split the data into training and testing sets
train_data = data[:n]  # n is the number of training samples
test_data = data[n:]   # remaining samples for testing

# Create and fit the ARIMA model
model = ARIMA(train_data, order=(p, d, q))  # p, d, q are model parameters
fitted_model = model.fit()

# Perform forecasting
predictions = fitted_model.forecast(steps=len(test_data))

# Evaluate the model
mse = np.mean((predictions - test_data)**2)
mae = np.mean(np.abs(predictions - test_data))

# Visualize the results
plt.plot(train_data, label='Training data')
plt.plot(test_data, label='Actual data')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
