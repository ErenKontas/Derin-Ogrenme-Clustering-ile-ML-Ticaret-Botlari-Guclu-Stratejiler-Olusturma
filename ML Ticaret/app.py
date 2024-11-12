import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import joblib


# Load data and update column names
df = pd.read_csv('BTC-Hourly.csv')
df.columns = df.columns.str.replace(r'[\s\.]', '_', regex=True)

# Select dependent and independent variables
x = df[["open", "high", "low", "close", "Volume_BTC", "Volume_USD"]]

# Preprocessing (StandardScaler)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ["open", "high", "low", "close", "Volume_BTC", "Volume_USD"])
    ]
)

# Streamlit application
def fiyat_pred(open, high, low, close, Volume_BTC, Volume_USD):
    input_data = pd.DataFrame({
        'open': [open],
        'high': [high],
        'low': [low],
        'close': [close],
        'Volume_BTC': [Volume_BTC],
        'Volume_USD': [Volume_USD]
    })
    input_data_transformed = preprocessor.fit_transform(input_data)

    model = joblib.load('ML.pkl')

    prediction = model.predict(input_data_transformed)
    return float(prediction[0])

# Streamlit interface
def main():
    st.title("Prediction Model")
    st.write("Enter Input Data")
    
    open = st.slider('Open', float(df['open'].min()), float(df['open'].max()))
    high = st.slider('High', float(df['high'].min()), float(df['high'].max()))
    low = st.slider('Low', float(df['low'].min()), float(df['low'].max()))
    close = st.slider('Close', float(df['close'].min()), float(df['close'].max()))
    Volume_BTC = st.slider('Volume BTC', float(df['Volume_BTC'].min()), float(df['Volume_BTC'].max()))
    Volume_USD = st.slider('Volume USD', float(df['Volume_USD'].min()), float(df['Volume_USD'].max()))
    
    if st.button('Predict'):
        fiyat = fiyat_pred(open, high, low, close, Volume_BTC, Volume_USD)
        st.write(f'The predicted price is: {fiyat:.2f}')

if __name__ == '__main__':
    main()

