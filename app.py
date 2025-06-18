import streamlit as st
import numpy as np
from openai import OpenAI

# ✅ MUST BE FIRST Streamlit command
st.set_page_config(page_title="Portfolio AI Assistant", layout="wide")

# ✅ Use new OpenAI client (for v1.x+)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- PPO model stub functions ---
def load_pretrained_ppo_model():
    st.info("🔁 PPO model loaded (stub).")
    return "ppo_model_stub"

def get_model_output(model, state=None):
    return np.random.rand(5)

# --- GPT-4 Chat UI ---
st.title("💹 Crypto Portfolio AI Assistant 🤖")
st.write("Ask me anything about your portfolio. I’ll help you decide what to buy, sell, or hold.")

user_input = st.text_input("Your question about portfolio:", "")

if user_input:
    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a portfolio assistant using FinRL-style logic."},
                {"role": "user", "content": user_input}
            ]
        )
        st.success(response.choices[0].message.content)

# --- PPO Section ---
with st.expander("🤖 Run PPO Model"):
    if st.button("Load PPO Model"):
        model = load_pretrained_ppo_model()
        st.session_state['ppo_model'] = model

    if st.session_state.get("ppo_model"):
        output = get_model_output(st.session_state["ppo_model"])
        st.write("📤 PPO Output:", output)

# --- Sidebar: IDX toggle ---
st.sidebar.markdown("### Options")
use_idx = st.sidebar.checkbox("Enable Index Mode")

if use_idx:
    st.sidebar.write("Index Mode is ON (stub behavior)")

if st.button("📥 Download Results (Stub)"):
    st.success("Download triggered (stub, no file yet).")
