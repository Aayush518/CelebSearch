import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

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


second_prompt = PromptTemplate(
   input_variables = ['person'],
    template ="when was {person} born?.",
)

chain2 = LLMChain(llm=llm, prompt=second_prompt, verbose=True)

parent_chain = SequentialChain(chains=[chain, chain2], input_variables=['name'], output_variables=['person', 'dob'], verbose=True)


if input_text:
    st.write(parent_chain({'name': input_text}))