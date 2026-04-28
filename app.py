# ============================================================
# Stock Analytics and Portfolio Dashboard App
# Streamlit App for GitHub + Streamlit Cloud
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Stock Analytics and Portfolio Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Analytics and Portfolio Dashboard")
st.write(
    """
    This app analyzes one individual stock and one five-stock portfolio using real market data 
    from Yahoo Finance. It calculates moving averages, RSI, volatility, trading signals, 
    portfolio returns, benchmark comparison, and Sharpe ratio.
    """
)

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

@st.cache_data
def download_stock_data(symbol, period):
    data = yf.download(symbol, period=period, interval="1d", auto_adjust=True)
    return data


@st.cache_data
def download_portfolio_data(symbols, period):
    data = yf.download(symbols, period=period, interval="1d", auto_adjust=True)["Close"]
    return data


def calculate_rsi(prices, window=14):
    delta = prices.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def classify_trend(current_price, ma20, ma50):
    if current_price > ma20 > ma50:
        return "Strong Uptrend"
    elif current_price < ma20 < ma50:
        return "Strong Downtrend"
    else:
        return "Mixed Trend"


def classify_rsi(rsi):
    if rsi > 70:
        return "Overbought - Possible Sell Signal"
    elif rsi < 30:
        return "Oversold - Possible Buy Signal"
    else:
        return "Neutral"


def classify_volatility(volatility):
    if volatility > 0.40:
        return "High"
    elif volatility >= 0.25:
        return "Medium"
    else:
        return "Low"


def make_recommendation(trend, rsi, volatility_level):
    if trend == "Strong Uptrend" and rsi < 70 and volatility_level != "High":
        return "Buy", "The stock is in a strong uptrend, RSI is not overbought, and volatility is not too high."
    elif trend == "Strong Downtrend" or rsi > 70:
        return "Sell", "The stock is either in a downtrend or appears overbought based on RSI."
    else:
        return "Hold", "The signals are mixed, so holding may be the safest decision."


# ------------------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------------------

st.sidebar.header("User Inputs")

individual_stock = st.sidebar.text_input(
    "Choose one stock for individual analysis:",
    value="AAPL"
).upper()

st.sidebar.subheader("Portfolio Setup")

stock1 = st.sidebar.text_input("Stock 1", value="AAPL").upper()
stock2 = st.sidebar.text_input("Stock 2", value="MSFT").upper()
stock3 = st.sidebar.text_input("Stock 3", value="NVDA").upper()
stock4 = st.sidebar.text_input("Stock 4", value="AMZN").upper()
stock5 = st.sidebar.text_input("Stock 5", value="GOOGL").upper()

portfolio_stocks = [stock1, stock2, stock3, stock4, stock5]

st.sidebar.subheader("Portfolio Weights")

weight1 = st.sidebar.number_input("Weight for Stock 1", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
weight2 = st.sidebar.number_input("Weight for Stock 2", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
weight3 = st.sidebar.number_input("Weight for Stock 3", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
weight4 = st.sidebar.number_input("Weight for Stock 4", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
weight5 = st.sidebar.number_input("Weight for Stock 5", min_value=0.0, max_value=1.0, value=0.15, step=0.01)

weights = np.array([weight1, weight2, weight3, weight4, weight5])

benchmark_symbol = st.sidebar.text_input(
    "Benchmark ETF:",
    value="SPY"
).upper()

run_analysis = st.sidebar.button("Run Analysis")

# ------------------------------------------------------------
# Run App
# ------------------------------------------------------------

if run_analysis:

    if round(weights.sum(), 2) != 1.00:
        st.error(f"Your portfolio weights must add up to 1.00. Current total: {weights.sum():.2f}")
        st.stop()

    # ========================================================
    # Part 1: Individual Stock Analysis
    # ========================================================

    st.header("Part 1: Individual Stock Analysis")

    stock_data = download_stock_data(individual_stock, "6mo")

    if stock_data.empty:
        st.error("No data found for the individual stock. Please check the ticker symbol.")
        st.stop()

    stock_data = stock_data[["Close"]].dropna()
    stock_data.columns = ["Close"]

    stock_data["20-Day Moving Average"] = stock_data["Close"].rolling(window=20).mean()
    stock_data["50-Day Moving Average"] = stock_data["Close"].rolling(window=50).mean()
    stock_data["Daily Return"] = stock_data["Close"].pct_change()
    stock_data["RSI"] = calculate_rsi(stock_data["Close"], 14)
    stock_data["20-Day Annualized Volatility"] = stock_data["Daily Return"].rolling(window=20).std() * np.sqrt(252)

    current_price = stock_data["Close"].iloc[-1]
    ma20 = stock_data["20-Day Moving Average"].iloc[-1]
    ma50 = stock_data["50-Day Moving Average"].iloc[-1]
    current_rsi = stock_data["RSI"].iloc[-1]
    current_volatility = stock_data["20-Day Annualized Volatility"].iloc[-1]

    trend = classify_trend(current_price, ma20, ma50)
    rsi_signal = classify_rsi(current_rsi)
    volatility_level = classify_volatility(current_volatility)
    recommendation, explanation = make_recommendation(trend, current_rsi, volatility_level)

    st.subheader(f"Stock Selected: {individual_stock}")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("20-Day MA", f"${ma20:.2f}")
    col3.metric("50-Day MA", f"${ma50:.2f}")
    col4.metric("RSI", f"{current_rsi:.2f}")

    stock_summary = pd.DataFrame({
        "Metric": [
            "Current Price",
            "20-Day Moving Average",
            "50-Day Moving Average",
            "Trend",
            "14-Day RSI",
            "RSI Signal",
            "20-Day Annualized Volatility",
            "Volatility Level",
            "Final Recommendation"
        ],
        "Result": [
            f"${current_price:.2f}",
            f"${ma20:.2f}",
            f"${ma50:.2f}",
            trend,
            f"{current_rsi:.2f}",
            rsi_signal,
            f"{current_volatility:.2%}",
            volatility_level,
            recommendation
        ]
    })

    st.subheader("Individual Stock Summary")
    st.dataframe(stock_summary, use_container_width=True)

    st.subheader("Price Trend Chart")

    fig1, ax1 = plt.subplots()
    ax1.plot(stock_data.index, stock_data["Close"], label="Closing Price")
    ax1.plot(stock_data.index, stock_data["20-Day Moving Average"], label="20-Day Moving Average")
    ax1.plot(stock_data.index, stock_data["50-Day Moving Average"], label="50-Day Moving Average")
    ax1.set_title(f"{individual_stock} Price Trend Analysis")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    st.subheader("RSI Momentum Chart")

    fig2, ax2 = plt.subplots()
    ax2.plot(stock_data.index, stock_data["RSI"], label="14-Day RSI")
    ax2.axhline(70, linestyle="--", label="Overbought Level")
    ax2.axhline(30, linestyle="--", label="Oversold Level")
    ax2.set_title(f"{individual_stock} RSI Analysis")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("RSI")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    st.subheader("Volatility Chart")

    fig3, ax3 = plt.subplots()
    ax3.plot(stock_data.index, stock_data["20-Day Annualized Volatility"], label="20-Day Annualized Volatility")
    ax3.set_title(f"{individual_stock} Volatility Analysis")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Annualized Volatility")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    st.subheader("Trading Recommendation")
    st.success(f"Recommendation: {recommendation}")
    st.write(explanation)

    # ========================================================
    # Part 2: Portfolio Performance Dashboard
    # ========================================================

    st.header("Part 2: Portfolio Performance Dashboard")

    st.write("Selected Portfolio Stocks:")
    st.write(portfolio_stocks)

    allocation_table = pd.DataFrame({
        "Stock": portfolio_stocks,
        "Weight": weights
    })

    st.subheader("Portfolio Allocation")
    st.dataframe(allocation_table, use_container_width=True)

    portfolio_data = download_portfolio_data(portfolio_stocks, "1y")
    benchmark_data = download_stock_data(benchmark_symbol, "1y")

    if portfolio_data.empty or benchmark_data.empty:
        st.error("Portfolio or benchmark data could not be downloaded. Please check your ticker symbols.")
        st.stop()

    benchmark_data = benchmark_data[["Close"]].dropna()
    benchmark_data.columns = ["Benchmark Close"]

    portfolio_data = portfolio_data.dropna()

    stock_returns = portfolio_data.pct_change().dropna()
    portfolio_returns = stock_returns.dot(weights)

    benchmark_returns = benchmark_data["Benchmark Close"].pct_change().dropna()

    combined_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    combined_returns.columns = ["Portfolio Return", "Benchmark Return"]

    portfolio_total_return = (1 + combined_returns["Portfolio Return"]).prod() - 1
    benchmark_total_return = (1 + combined_returns["Benchmark Return"]).prod() - 1
    performance_difference = portfolio_total_return - benchmark_total_return

    portfolio_volatility = combined_returns["Portfolio Return"].std() * np.sqrt(252)
    benchmark_volatility = combined_returns["Benchmark Return"].std() * np.sqrt(252)

    portfolio_sharpe = (combined_returns["Portfolio Return"].mean() * 252) / portfolio_volatility
    benchmark_sharpe = (combined_returns["Benchmark Return"].mean() * 252) / benchmark_volatility

    portfolio_summary = pd.DataFrame({
        "Metric": [
            "Portfolio Total Return",
            "Benchmark Total Return",
            "Outperformance / Underperformance",
            "Portfolio Annualized Volatility",
            "Benchmark Annualized Volatility",
            "Portfolio Sharpe Ratio",
            "Benchmark Sharpe Ratio"
        ],
        "Result": [
            f"{portfolio_total_return:.2%}",
            f"{benchmark_total_return:.2%}",
            f"{performance_difference:.2%}",
            f"{portfolio_volatility:.2%}",
            f"{benchmark_volatility:.2%}",
            f"{portfolio_sharpe:.2f}",
            f"{benchmark_sharpe:.2f}"
        ]
    })

    st.subheader("Portfolio Performance Summary")
    st.dataframe(portfolio_summary, use_container_width=True)

    col5, col6, col7 = st.columns(3)

    col5.metric("Portfolio Return", f"{portfolio_total_return:.2%}")
    col6.metric("Benchmark Return", f"{benchmark_total_return:.2%}")
    col7.metric("Difference", f"{performance_difference:.2%}")

    st.subheader("Portfolio vs. Benchmark Growth of $1")

    cumulative_portfolio = (1 + combined_returns["Portfolio Return"]).cumprod()
    cumulative_benchmark = (1 + combined_returns["Benchmark Return"]).cumprod()

    fig4, ax4 = plt.subplots()
    ax4.plot(cumulative_portfolio.index, cumulative_portfolio, label="Portfolio")
    ax4.plot(cumulative_benchmark.index, cumulative_benchmark, label=f"Benchmark: {benchmark_symbol}")
    ax4.set_title("Portfolio Performance vs. Benchmark")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Growth of $1")
    ax4.legend()
    ax4.grid(True)
    st.pyplot(fig4)

    st.subheader("Portfolio Allocation Chart")

    fig5, ax5 = plt.subplots()
    ax5.pie(weights, labels=portfolio_stocks, autopct="%1.1f%%")
    ax5.set_title("Portfolio Allocation")
    st.pyplot(fig5)

    st.subheader("Risk and Return Comparison")

    risk_return_table = pd.DataFrame({
        "Investment": ["Portfolio", benchmark_symbol],
        "Total Return": [portfolio_total_return, benchmark_total_return],
        "Annualized Volatility": [portfolio_volatility, benchmark_volatility],
        "Sharpe Ratio": [portfolio_sharpe, benchmark_sharpe]
    })

    st.dataframe(risk_return_table, use_container_width=True)

    fig6, ax6 = plt.subplots()
    ax6.bar(risk_return_table["Investment"], risk_return_table["Total Return"])
    ax6.set_title("Total Return Comparison")
    ax6.set_ylabel("Total Return")
    ax6.grid(True)
    st.pyplot(fig6)

    fig7, ax7 = plt.subplots()
    ax7.bar(risk_return_table["Investment"], risk_return_table["Annualized Volatility"])
    ax7.set_title("Annualized Volatility Comparison")
    ax7.set_ylabel("Annualized Volatility")
    ax7.grid(True)
    st.pyplot(fig7)

    fig8, ax8 = plt.subplots()
    ax8.bar(risk_return_table["Investment"], risk_return_table["Sharpe Ratio"])
    ax8.set_title("Sharpe Ratio Comparison")
    ax8.set_ylabel("Sharpe Ratio")
    ax8.grid(True)
    st.pyplot(fig8)

    # ========================================================
    # Written Interpretation
    # ========================================================

    st.header("Written Interpretation")

    if portfolio_total_return > benchmark_total_return:
        outperform_text = "The portfolio outperformed the benchmark."
    else:
        outperform_text = "The portfolio underperformed the benchmark."

    if portfolio_volatility > benchmark_volatility:
        risk_text = "The portfolio was more risky than the benchmark based on annualized volatility."
    else:
        risk_text = "The portfolio was less risky than the benchmark based on annualized volatility."

    if portfolio_sharpe > benchmark_sharpe:
        sharpe_text = "The portfolio was more efficient than the benchmark because it had a higher Sharpe ratio."
    else:
        sharpe_text = "The portfolio was less efficient than the benchmark because it had a lower Sharpe ratio."

    st.write(f"""
    The individual stock analysis for **{individual_stock}** suggests a **{recommendation}** recommendation. 
    The trend analysis showed a **{trend}**. The RSI was **{current_rsi:.2f}**, which indicates **{rsi_signal}**. 
    The volatility level was **{volatility_level}**, with annualized volatility of **{current_volatility:.2%}**.

    For the portfolio, the total return was **{portfolio_total_return:.2%}**, while the benchmark return was 
    **{benchmark_total_return:.2%}**. {outperform_text}

    In terms of risk, the portfolio annualized volatility was **{portfolio_volatility:.2%}**, compared to the 
    benchmark volatility of **{benchmark_volatility:.2%}**. {risk_text}

    The portfolio Sharpe ratio was **{portfolio_sharpe:.2f}**, compared to the benchmark Sharpe ratio of 
    **{benchmark_sharpe:.2f}**. {sharpe_text}

    Overall, the portfolio should be evaluated by looking at both return and risk. A higher return is good, 
    but the Sharpe ratio helps show whether the return was worth the amount of risk taken.
    """)

else:
    st.info("Use the sidebar to enter your stocks and weights, then click **Run Analysis**.")
