import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from openai import OpenAI
import requests
import ccxt
import yfinance as yf
import plotly.express as px

# Set page config
st.set_page_config(page_title="Yeildera Portfolio AI Assistant", layout="wide")

# OpenAI client setup
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("OpenAI API key not found in Streamlit secrets. Please configure it.")
    st.stop()

# CoinMarketCap API key (store in secrets)
try:
    CMC_API_KEY = st.secrets["CMC_API_KEY"]
except KeyError:
    st.error("CoinMarketCap API key not found in Streamlit secrets. Sign up at https://coinmarketcap.com/api/.")
    st.stop()

# --- Sidebar: Toggle IDX vs Crypto ---
st.sidebar.markdown("### Options")
use_idx = st.sidebar.checkbox("🇮🇩 Use IDX Mode")

if use_idx:
    asset_list = ["BBCA.JK", "TLKM.JK", "BMRI.JK"]
else:
    asset_list = ["bitcoin", "ethereum", "solana", "cardano", "polkadot", "chainlink", "litecoin"]

st.sidebar.write("Assets:", ", ".join(asset_list))

# --- LIVE DATA FUNCTIONS ---

def get_live_crypto_data(coin_ids, days=7):
    # Map coin IDs to CoinMarketCap symbols
    cmc_symbols = {
        "bitcoin": "BTC",
        "ethereum": "ETH",
        "solana": "SOL",
        "cardano": "ADA",
        "polkadot": "DOT",
        "chainlink": "LINK",
        "litecoin": "LTC"
    }
    symbols = ",".join(cmc_symbols[coin.lower()] for coin in coin_ids)
    
    try:
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
        headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
        params = {"symbol": symbols, "convert": "USD"}
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            prices = {}
            for coin in coin_ids:
                symbol = cmc_symbols[coin.lower()]
                price = data["data"][symbol]["quote"]["USD"]["price"]
                # Simulate historical data (free tier lacks historical endpoint)
                prices[coin] = [price] * days  # Placeholder: Use ccxt for historical data
            df = pd.DataFrame(prices)
            df.index = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
            return df
        else:
            raise Exception(f"CoinMarketCap error: {response.json().get('status', {}).get('error_message', 'Unknown error')}")
    except Exception as e:
        st.warning(f"CoinMarketCap failed: {e}. Falling back to ccxt.")
        try:
            exchange = ccxt.binance()
            prices = {}
            for coin in coin_ids:
                symbol = f"{cmc_symbols[coin.lower()]}/USDT"
                data = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=days)
                prices[coin] = [row[4] for row in data]  # Close price
            df = pd.DataFrame(prices)
            df.index = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
            return df
        except Exception as e:
            st.error(f"ccxt fallback failed: {e}")
            return None

def get_live_idx_data(tickers, start="2023-01-01", end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    try:
        df = yf.download(tickers, start=start, end=end)["Adj Close"]
        if df.empty:
            st.error("No data returned for IDX tickers.")
            return None
        return df.dropna()
    except Exception as e:
        st.error(f"Failed to fetch IDX data: {e}")
        return None

# --- Optimizer Logic ---
def calculate_metrics(returns, inflation_rate=0):
    if returns.empty:
        raise ValueError("Returns data is empty.")
    mean_returns = returns.mean() * 252 - inflation_rate / 100
    cov_matrix = returns.cov() * 252
    return mean_returns, cov_matrix

def generate_portfolios(n, mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    results = np.zeros((3, n))
    weights_record = []
    for i in range(n):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        ret = np.sum(weights * mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = ret / vol if vol != 0 else 0
        results[0, i] = ret
        results[1, i] = vol
        results[2, i] = sharpe
    return results, weights_record

def get_model_output(coin_ids, days=7):
    if use_idx:
        df = get_live_idx_data(coin_ids)
    else:
        df = get_live_crypto_data(coin_ids, days)
    if df is None or df.empty:
        st.error("Failed to retrieve valid market data.")
        return None
    returns = df.pct_change().dropna()
    if returns.empty:
        st.error("No valid returns calculated. Check data quality.")
        return None
    try:
        mean_returns, cov_matrix = calculate_metrics(returns)
        results, weights = generate_portfolios(100, mean_returns, cov_matrix)
        best_idx = np.argmax(results[2])
        return dict(zip(coin_ids, weights[best_idx]))
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        return None

# --- PPO Placeholder ---
# from stable_baselines3 import PPO
# from finrl.env_stock_trading import StockTradingEnv
# def get_ppo_output(df, coin_ids):
#     env = StockTradingEnv(df=df, initial_amount=10000)
#     model = PPO.load("ppo_yeildera")
#     obs = env.reset()
#     action, _ = model.predict(obs)
#     return dict(zip(coin_ids, action / np.sum(action)))

# --- Streamlit UI ---
st.title("💹 Yeildera Portfolio AI Assistant 🤖")
st.write("Ask about your crypto or IDX portfolio. Powered by live market data and AI optimization.")

user_input = st.text_input("Your question about portfolio:", "")

if user_input:
    if "allocate" in user_input.lower() or "suggest" in user_input.lower() or "invest" in user_input.lower():
        with st.spinner("Optimizing portfolio using live data..."):
            alloc = get_model_output(asset_list)
            if alloc:
                alloc_str = ", ".join([f"{k}: {round(v*100, 2)}%" for k, v in alloc.items()])
                gpt_input = f"The optimized portfolio allocation is: {alloc_str}. Explain this allocation in simple terms for a retail investor."
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You're a helpful portfolio assistant for retail investors."},
                            {"role": "user", "content": gpt_input}
                        ]
                    )
                    st.success(response.choices[0].message.content)
                    st.info(f"📊 Allocation: {alloc_str}")
                    alloc_df = pd.DataFrame(list(alloc.items()), columns=["Asset", "Weight"])
                    fig = px.pie(alloc_df, values="Weight", names="Asset", title="Portfolio Allocation")
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Failed to get GPT response: {e}")
            else:
                st.error("Unable to generate portfolio allocation.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You're a helpful portfolio assistant for retail investors."},
                        {"role": "user", "content": user_input}
                    ]
                )
                st.success(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Failed to get GPT response: {e}")

# Disclaimer
st.markdown("---")
st.caption("⚠️ This is not financial advice. Always consult a professional before investing.")
