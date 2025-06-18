
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from openai import OpenAI
from pycoingecko import CoinGeckoAPI
import yfinance as yf

# ✅ Set page config first
st.set_page_config(page_title="Portfolio AI Assistant", layout="wide")

# ✅ OpenAI client setup
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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
    cg = CoinGeckoAPI()
    prices = {}
    for coin in coin_ids:
        try:
            data = cg.get_coin_market_chart_by_id(id=coin, vs_currency='usd', days=days)
            prices[coin] = [price[1] for price in data['prices']]
        except Exception as e:
            st.error(f"Failed to fetch {coin}: {e}")
            prices[coin] = [np.nan] * days
    df = pd.DataFrame(prices)
    df.index = pd.date_range(end=datetime.now(), periods=len(df))
    return df

def get_live_idx_data(tickers, start="2023-01-01", end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(tickers, start=start, end=end)["Adj Close"]
    return df.dropna()

# --- Optimizer logic ---
def calculate_metrics(returns, inflation_rate=0):
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
    returns = df.pct_change().dropna()
    mean_returns, cov_matrix = calculate_metrics(returns)
    results, weights = generate_portfolios(100, mean_returns, cov_matrix)
    best_idx = np.argmax(results[2])
    return dict(zip(coin_ids, weights[best_idx]))

# --- GPT Chat UI ---
st.title("💹 Portfolio AI Assistant 🤖")
st.write("Ask anything about your crypto or IDX portfolio.")

user_input = st.text_input("Your question about portfolio:", "")

if user_input:
    if "allocate" in user_input.lower() or "suggest" in user_input.lower() or "invest" in user_input.lower():
        with st.spinner("Optimizing portfolio using live data..."):
            alloc = get_model_output(asset_list)
            alloc_str = ", ".join([f"{k}: {round(v*100,2)}%" for k,v in alloc.items()])
            gpt_input = f"The optimized portfolio allocation is: {alloc_str}"
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You're a helpful portfolio assistant."},
                    {"role": "user", "content": gpt_input}
                ]
            )
            st.success(response.choices[0].message.content)
            st.info("📊 Allocation:
" + gpt_input)
    else:
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You're a helpful portfolio assistant."},
                    {"role": "user", "content": user_input}
                ]
            )
            st.success(response.choices[0].message.content)
