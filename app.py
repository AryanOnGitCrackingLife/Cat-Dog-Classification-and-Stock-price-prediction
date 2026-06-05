from flask import Flask, render_template, request
import cv2
import numpy as np
import os

from tensorflow.keras.models import load_model 
import joblib
import yfinance as yf

app = Flask(__name__)

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CNN model
CNN_MODEL_PATH = os.path.join(BASE_DIR, "Cat_dog", "models", "cnn.h5")
cnn_model = load_model(CNN_MODEL_PATH)

# Stock models
LR_MODEL_PATH = os.path.join(BASE_DIR, "stock", "stock_lr.pkl")
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "stock", "lstm.h5")
SCALER_PATH = os.path.join(BASE_DIR, "stock", "scaler.pkl")

lr_model = joblib.load(LR_MODEL_PATH)
lstm_model = load_model(LSTM_MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# -------------------------------------------------
# ROUTE 1: CATS vs DOGS
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

            if img is not None:
                img = cv2.resize(img, (64, 64))
                img = img / 255.0
                img = img.reshape(1, 64, 64, 3)

                prediction = cnn_model.predict(img)[0][0]
                result = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"

    return render_template("index.html", result=result)

# -------------------------------------------------
# ROUTE 2: STOCK PREDICTION
# -------------------------------------------------
@app.route("/stock", methods=["GET", "POST"])
def stock():
    lr_price = None
    lstm_price = None

    if request.method == "POST":
        years = int(request.form.get("years"))
        future_days = years * 252  # trading days

        # Download stock data
        data = yf.download("AAPL", start="2015-01-01")
        data["Days"] = np.arange(len(data))

        # ---- Linear Regression Prediction ----
        last_day = data["Days"].iloc[-1]
        future_day = last_day + future_days
        lr_price = float(lr_model.predict([[future_day]])[0])

        # ---- LSTM Prediction ----
        close_prices = data["Close"].values[-60:].reshape(-1, 1)
        scaled_input = scaler.transform(close_prices)
        scaled_input = scaled_input.reshape(1, 60, 1)

        lstm_scaled = lstm_model.predict(scaled_input)
        lstm_price = scaler.inverse_transform(lstm_scaled)[0][0]
        lstm_price = float(lstm_price)

    return render_template(
        "stock.html",
        lr_price=lr_price,
        lstm_price=lstm_price
    )

# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
