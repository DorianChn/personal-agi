"""Streamlit 对话界面。"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.client import get_client  # noqa: E402

st.set_page_config(page_title="Personal AGI", page_icon="🤖", layout="wide")
st.title("🤖 Personal AGI — 你的专属助手")

with open("config/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
gen = cfg.get("generation", {})
client = get_client(cfg)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("和你的专属模型聊聊…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            resp = client.chat(
                st.session_state.messages,
                temperature=gen.get("temperature", 0.7),
                top_p=gen.get("top_p", 0.9),
                max_tokens=gen.get("max_tokens", 1024),
            )
        st.markdown(resp)
    st.session_state.messages.append({"role": "assistant", "content": resp})
