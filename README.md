# STEP 1: Import Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from scipy.optimize import differential_evolution
import streamlit as st
import matplotlib.pyplot as plt

# STEP 2: Simulate Inflow, Demand, Rainfall
np.random.seed(42)
days = 365
inflow = np.random.normal(loc=100, scale=20, size=days)
demand = np.random.normal(loc=90, scale=15, size=days)
rainfall = np.random.normal(loc=50, scale=10, size=days)

data = pd.DataFrame({
    'inflow': inflow,
    'demand': demand,
    'rainfall': rainfall
})

# STEP 3: Normalize and Prepare Sequences
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, :2])  # Predict inflow and demand
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))  # Shape: (samples, 10, 3)

# STEP 4: Build and Train LSTM Model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    Dense(2)  # Output: inflow and demand
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, verbose=0)

# STEP 5: Forecast the Next 30 Days
forecast_scaled = []
last_seq = X[-1:]  # Last known 10-day sequence

for _ in range(30):
    pred = model.predict(last_seq, verbose=0)[0]
    forecast_scaled.append(pred)
    rainfall_value = last_seq[0, -1, 2]
    new_input = np.array([pred[0], pred[1], rainfall_value])
    last_seq = np.append(last_seq[:, 1:, :], [[new_input]], axis=1)

dummy_rainfall = np.full((30, 1), rainfall.mean())
full_forecast_scaled = np.hstack([forecast_scaled, dummy_rainfall])
forecast = scaler.inverse_transform(full_forecast_scaled)[:, :2]
forecast_inflow = forecast[:, 0]
forecast_demand = forecast[:, 1]

# STEP 6: Genetic Algorithm Optimization for Water Release
def fitness(schedule):
    reservoir = 1000
    penalty = 0
    for i in range(len(schedule)):
        reservoir += forecast_inflow[i] - schedule[i]
        if schedule[i] < forecast_demand[i]:
            penalty += (forecast_demand[i] - schedule[i]) ** 2
        if reservoir > 1200:
            penalty += (reservoir - 1200) ** 2
        if reservoir < 800:
            penalty += (800 - reservoir) ** 2
    return penalty

bounds = [(70, 120)] * 30
result = differential_evolution(fitness, bounds, maxiter=30, disp=False)
release_schedule = result.x

# STEP 7: Streamlit Dashboard
st.set_page_config(page_title="Smart Reservoir Dashboard", layout="wide")
st.title(" Smart Reservoir Management System")

tab1, tab2, tab3 = st.tabs(["Forecasted Data", "Optimized Release", "Reservoir Simulation"])

with tab1:
    st.subheader(" Forecasted Inflow and Demand (Next 30 Days)")
    df_forecast = pd.DataFrame({
        'Day': np.arange(1, 31),
        'Forecasted Inflow': forecast_inflow,
        'Forecasted Demand': forecast_demand
    })
    st.line_chart(df_forecast.set_index('Day'))

with tab2:
    st.subheader(" Optimized Water Release Schedule")
    df_release = pd.DataFrame({
        'Day': np.arange(1, 31),
        'Release': release_schedule
    })
    st.line_chart(df_release.set_index('Day'))
    st.write("First 5 Days of Optimized Release:")
    st.write(df_release.head())

with tab3:
    st.subheader(" Simulated Reservoir Levels Over 30 Days")
    reservoir = [1000]
    for i in range(30):
        reservoir.append(reservoir[-1] + forecast_inflow[i] - release_schedule[i])
    df_reservoir = pd.DataFrame({'Day': np.arange(1, 31), 'Reservoir Level': reservoir[1:]})
    st.line_chart(df_reservoir.set_index('Day'))

st.success(" Optimization complete! Water usage planned efficiently with LSTM + GA ")
