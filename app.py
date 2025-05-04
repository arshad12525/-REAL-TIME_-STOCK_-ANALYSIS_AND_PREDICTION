import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score
import plotly.graph_objs as go
from datetime import timedelta

# Page config
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Sidebar Navigation
page = st.sidebar.selectbox("Choose Page", ["Stock Dashboard", "Stock Price Prediction"])

# Sliding window function
def create_features(data, window_size=5):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Technical Indicator Functions
def add_technical_indicators(df):
    # Adding moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # Adding Relative Strength Index (RSI)
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# Handle Indian stocks (.NS if missing)
def format_symbol(symbol):
    indian_stocks = ['RELIANCE', 'TCS', 'INFY', 'ZOMATO']
    if symbol.upper() in indian_stocks and not symbol.upper().endswith(".NS"):
        return symbol.upper() + ".NS"
    return symbol.upper()

# Stock Dashboard
if page == "Stock Dashboard":
    st.title("📈 Real-Time Stock Market Dashboard")

    popular_stocks = {
        "Apple (AAPL)": "AAPL",
        "Google (GOOGL)": "GOOGL",  # Changed from GOOG to GOOGL
        "Tesla (TSLA)": "TSLA",
        "Reliance (RELIANCE.NS)": "RELIANCE.NS",
        "TCS (TCS.NS)": "TCS.NS",
        "Infosys (INFY.NS)": "INFY.NS",
        "Zomato (ZOMATO.NS)": "ZOMATO.NS",
    }

    selected_label = st.selectbox("Select Popular Stock", options=list(popular_stocks.keys()))
    custom_input = st.text_input("Or enter custom Yahoo Finance symbol (e.g., INFY.NS)", popular_stocks[selected_label])

    stock_symbol = format_symbol(custom_input.strip())
    st.write("---")

    if stock_symbol:
        try:
            ticker = yf.Ticker(stock_symbol)

            # Select date range
            st.subheader("Select Date Range for Analysis 📅")
            start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
            end_date = st.date_input("End Date", value=pd.to_datetime("today"))

            hist = ticker.history(start=start_date, end=end_date)

            if hist.empty:
                st.warning("⚠️ No data available. Please check the symbol or date range.")
            else:
                # Apply technical indicators
                hist = add_technical_indicators(hist)

                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[0]
                change = current_price - prev_price
                pct_change = (change / prev_price) * 100

                price_color = "green" if change > 0 else "red"

                st.markdown(f"""
                    <div style="background-color: #1e1e1e; padding: 15px; border-left: 6px solid {price_color}; border-radius: 10px;">
                        <h2 style="color:white;">Live Price: ₹{current_price:.2f} ({change:+.2f}, {pct_change:+.2f}%)</h2>
                    </div>
                """, unsafe_allow_html=True)

                # Company Info
                info = ticker.info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Company", info.get("shortName", "N/A"))
                    st.metric("Sector", info.get("sector", "N/A"))
                with col2:
                    st.metric("Market Cap", f"{info.get('marketCap', 0):,}")
                    st.metric("P/E Ratio", info.get("trailingPE", "N/A"))
                with col3:
                    st.metric("Dividend Yeild", info.get("dividendYield"))

                website = info.get("website", "")
                if website:
                    st.markdown(f"[🌐 Company Website]({website})")

                # 🎯 Table: Open, High, Low, Close
                st.write("### Historical Stock Data Table 📋")
                st.dataframe(hist[['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'MA20', 'RSI']].tail())

                # 📈 Closing Prices Chart
                if not hist.empty:
                    st.write("### Closing Prices Chart 📊")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name="Closing Price", line=dict(color="cyan")))
                    fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Price (INR)")
                    st.plotly_chart(fig, use_container_width=True)

                # 📊 Basic EDA
                st.write("### 📊 Basic Data Insights:")
                st.write(f"**Highest Closing Price:** ₹{hist['Close'].max():.2f}")
                st.write(f"**Lowest Closing Price:** ₹{hist['Close'].min():.2f}")
                st.write(f"**Average Volume:** {hist['Volume'].mean():,.0f}")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# Stock Price Prediction
elif page == "Stock Price Prediction":
    st.title("🔮 Stock Price Prediction")

    stock_symbol = st.text_input("Enter Stock Symbol (e.g., INFY, TCS, AAPL)")
    prediction_days = st.slider("Days to predict", 1, 30, 5)

    if stock_symbol:
        stock_symbol = format_symbol(stock_symbol)
        ticker = yf.Ticker(stock_symbol)
        data = ticker.history(period="5y", interval="1d")

        if data.empty:
            st.warning("⚠️ No data available for this stock.")
        else:
            st.write("Last available data:")
            st.dataframe(data.tail())

            closing_prices = data['Close'].values

            # Normalize the closing prices
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(closing_prices.reshape(-1, 1))

            # Create feature sets
            X, y = create_features(scaled_data, window_size=5)
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

            # Reshape for LSTM input
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # Build LSTM model
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dense(1)
            ])

            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

            # Predict stock prices
            y_pred = model.predict(X_test)

            # Rescale predictions and actual values
            y_pred_rescaled = scaler.inverse_transform(y_pred)
            y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Show model performance
            mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
            st.success(f"Model MSE on test data: {mse:.2f}")

            # R² Score
            r2 = r2_score(y_test, y_pred)
            st.write(f"Model R² Score on test data: {r2:.2f}")

            # Predict future prices
            last_window = scaled_data[-5:]
            future_predictions = []
            for _ in range(prediction_days):
                next_pred = model.predict(last_window.reshape(1, 5, 1))[0][0]
                future_predictions.append(scaler.inverse_transform([[next_pred]])[0][0])
                last_window = np.append(last_window[1:], next_pred)

            # Dates
            last_date = data.index[-1]
            predicted_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days+1)]

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Historical Prices", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=predicted_dates, y=future_predictions, name="Predicted Prices", line=dict(color="red", dash="dot")))
            fig.update_layout(template="plotly_dark", title="Stock Price Prediction", xaxis_title="Date", yaxis_title="Price (INR)")
            st.plotly_chart(fig, use_container_width=True)

            # Show prediction table
            result_df = pd.DataFrame({
                "Date": predicted_dates,
                "Predicted Closing Price": future_predictions
            })
            st.write("### Predicted Future Prices 📅")
            st.dataframe(result_df)

            # Download predictions as CSV
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Predictions as CSV", data=csv, file_name=f"{stock_symbol}_predictions.csv", mime='text/csv')
