# Stock Price Prediction Web Application

This is a web application built using **Streamlit** for predicting stock prices using historical data. The app allows users to view real-time stock prices, perform technical analysis, and make predictions on future stock prices using an LSTM model.

# URL
https://real-timestockanalysisandprediction-zcfpl6wc8p6umrywddevrq.streamlit.app/

## Features
### 1. **Stock Dashboard**
   - View real-time stock market data for popular stocks.
   - Visualize the stock's **Closing Price**, **Open**, **High**, **Low**, and **Volume** over a selected time range.
   - Interactive charts for better analysis using **Plotly**.
   - Apply **Technical Indicators** like **Moving Averages (MA)** and **Relative Strength Index (RSI)**.
### 2. **Stock Price Prediction**
   - Predict future stock prices for any stock symbol (e.g., INFY, TCS, AAPL).
   - Use an **LSTM model** trained on historical stock data.
   - Display predicted stock prices for the next 1 to 30 days.
   - Option to download predictions as a **CSV** file.

## Installation
To run the project locally, follow these steps:
1. **Clone the repository:**
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction

2. **Create and activate a virtual environment:**
(You can skip this step if you already have a virtual environment set up.)
python -m venv .venv

- On Windows:
  ```
  .venv\Scripts\activate
  ```
- On MacOS/Linux:
  ```
  source .venv/bin/activate
  ```
3. **Install required libraries:**
You can install the required dependencies using `pip`:
pip install -r requirements.txt

4. **Run the app:**
To run the Streamlit app, use the following command:
streamlit run app.py

The app will launch in your default web browser.

## How to Use
1. **Stock Dashboard:**
- Select a stock from the dropdown or input a custom stock symbol.
- Choose a date range and see real-time data, including technical indicators and charts.
2. **Stock Price Prediction:**
- Enter a stock symbol (e.g., INFY, TCS, AAPL).
- Set the number of days for prediction (1-30 days).
- The app will display the predicted stock prices and a download link for the predictions in CSV format.

## Technical Indicators Used
- **Moving Averages (MA):** 5-day, 10-day, and 20-day moving averages.
- **Relative Strength Index (RSI):** A momentum oscillator that measures the speed and change of price movements.

## Technologies Used
- **Streamlit** for building the interactive web application.
- **yFinance** for fetching real-time and historical stock data.
- **TensorFlow (LSTM)** for stock price prediction.
- **Plotly** for interactive visualizations and stock charts.
- **Scikit-learn** for data preprocessing and model evaluation.

## Dependencies
To install the necessary libraries, you can use the following command:
pip install streamlit yfinance pandas numpy tensorflow scikit-learn plotly


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
[Arshad Shaikh]
