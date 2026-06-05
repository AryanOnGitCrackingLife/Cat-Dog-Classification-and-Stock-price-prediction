import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# 1. Download stock data
data = yf.download("AAPL", start="2015-01-01")
close_prices = data["Close"].values.reshape(-1, 1)

# 2. Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# 3. Create sequences
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

# 4. Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# 5. Train model
model.fit(X, y, epochs=5, batch_size=32)

# 6. Save model and scaler
model.save("lstm.h5")
joblib.dump(scaler, "scaler.pkl")

print("✅ LSTM model and scaler saved successfully")
