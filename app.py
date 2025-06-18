from flask import Flask, render_template, request
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)

# Load and preprocess data once at startup
def load_data():
    df = pd.read_csv('C:/Users/USER/Downloads/user-garbage_data_entity_202506101226.csv')  # Replace with your actual file
    df['date_time'] = pd.to_datetime(df['date_time'])
    df.set_index('date_time', inplace=True)
    ts = df['bin_level']  # Use only the univariate target
    ts = ts.resample('H').mean().interpolate()  # Resample if needed
    return ts

# Train SARIMAX model
def build_model(ts_series):
    model = SARIMAX(
        ts_series,
        order=(2, 1, 0),
        seasonal_order=(0, 0, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)
    return model_fit

# Load data and train model once
time_series = load_data()
sarima_model = build_model(time_series)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Optional: Get steps from form
        steps = int(request.form.get('steps', 100))

        forecast = sarima_model.forecast(steps=steps)
        forecast_rounded = forecast.round(2).tolist()

        return render_template('index.html', prediction=forecast_rounded)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)


