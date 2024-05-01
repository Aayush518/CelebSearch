import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

os.environ['OPENAI_API_KEY'] = openai_key

st.title("Celeb Search Results")
input_text = st.text_area("Search the topic you want")

first_prompt = PromptTemplate(
   input_variables = ['topic'],
    template ="Tell me about {name}.",
)

llm = OpenAI(temperature=0.8)

chain = LLMChain(llm=llm, prompt=first_prompt, verbose=True)




if input_text:
    st.write(chain(input_text)) 