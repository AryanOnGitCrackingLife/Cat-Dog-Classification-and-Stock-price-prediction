import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

data = yf.download("AAPL", start="2010-01-01")
data['Days'] = np.arange(len(data))

X = data[['Days']]
y = data['Close']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "stock_lr.pkl")
print("Linear Regression model saved")
