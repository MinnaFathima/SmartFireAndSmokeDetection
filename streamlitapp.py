import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
def load_data():
    if not os.path.exists("sensor_data_formatted.csv"):
        st.error("Dataset file not found! Please ensure 'sensor_data_formatted.csv' is in the same directory.")
        return None
    df = pd.read_csv("sensor_data_formatted.csv")
    return df

# Prepare data for training
def prepare_data(df):
    features = ["IR Sensor", "Smoke (PPM)", "Temperature (°C)", "O2 Level (%)", "Voltage (V)"]
    X = df[features].values
    y = df["Fire Risk Level"].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

# Build a simple Neural Network model
def build_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(X_train, y_train):
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    return model

# Predict fire probability
def predict_fire(model, scaler, new_data):
    new_data_scaled = scaler.transform(new_data)
    fire_prob = model.predict(new_data_scaled)[0][0] * 100
    return np.clip(fire_prob, 0, 100)

# Streamlit UI
st.title("🔥 Fire & Smoke Detection System")
df = load_data()

if df is not None:
    st.write("### Sample Sensor Data", df.head())
    
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    model = train_model(X_train, y_train)
    st.success("Model trained successfully!")
    
    # Plot Time Series Graphs
    st.write("### Sensor Data Trends Over Time")
    fig, axes = plt.subplots(5, 1, figsize=(10, 15))
    sensors = ["IR Sensor", "Smoke (PPM)", "Temperature (°C)", "O2 Level (%)", "Voltage (V)"]
    colors = ['black', 'red', 'green', 'purple', 'orange']
    
    for i, sensor in enumerate(sensors):
        axes[i].plot(df.index, df[sensor], color=colors[i], label=sensor)
        axes[i].set_title(f'{sensor} Over Time')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel(sensor)
        axes[i].legend()
    
    st.pyplot(fig)
    
    # User Input Form
    st.write("### Real-time Fire Prediction")
    with st.form(key='fire_form'):
        ir = st.selectbox("IR Sensor (0 - No Flame, 1 - Flickering/Constant, 2-high risk)", [0, 1])
        smoke = st.number_input("Smoke Level (PPM)", 0, 150, 90)
        temp = st.number_input("Temperature (°C)", 0, 100, 43)
        ##o2 = st.number_input("Oxygen Level (%)", 10.0, 21.0, 19.0)
        #voltage = st.number_input("Voltage (V)", 3.0, 5.5, 5.0)
        submit_button = st.form_submit_button("Predict Fire Probability")
    
    if submit_button:
        new_data = np.array([[ir, smoke, temp, o2, voltage]])
        fire_chance = predict_fire(model, scaler, new_data)
        st.write(f"🔥 Fire Probability: {fire_chance:.2f}%")
        
        if fire_chance > 70:
            st.error("⚠ Fire Detected! Activating LED & Buzzer! 🚨")
        elif fire_chance > 40:
            st.warning("⚠ Warning: Fire Risk Increasing!")
        else:
            st.success("✅ No fire detected.")