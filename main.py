import os
from constants import openai_key
from langchain.llms import OpenAI

import streamlit as st

os.environ['OPENAI_API_KEY'] = openai_key

st.title("LangChain Demo with OpenAI")
input_text = st.text_area("Search the topic you want")

OpenAI(temperature=0.8)