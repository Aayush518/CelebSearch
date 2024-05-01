import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st


os.environ['OPENAI_API_KEY'] = openai_key

st.title("Celeb Search Results")
input_text = st.text_area("Search the topic you want")

first_prompt = PromptTemplate(
   input_variables = ['topic'],
    template ="Tell me about {name}.",
)
person_memory = ConversationBufferMemory(input_variables=['person'], output_variables=['dob'], max_len=1)
dob_memory = ConversationBufferMemory(input_variables=['dob'], output_variables=['description'], max_len=1)
description_memory = ConversationBufferMemory(input_variables=['description'], output_variables=['person'], max_len=1)


llm = OpenAI(temperature=0.8)

chain = LLMChain(llm=llm, prompt=first_prompt, verbose=True, memory=person_memory, output_key='person')


second_prompt = PromptTemplate(
   input_variables = ['person'],
    template ="when was {person} born?.",
)

chain2 = LLMChain(llm=llm, prompt=second_prompt, verbose=True, memory=dob_memory, output_key='dob')

thrid_prompt = PromptTemplate(
   input_variables = ['dob'],
    template =" Mention 5 major events happened around {dob} in the world.",
)

chain3 = LLMChain(llm=llm, prompt=thrid_prompt, verbose=True,output_key='description', memory=description_memory )


parent_chain = SequentialChain(chains=[chain, chain2, chain3], input_variables=['name'], output_variables=['person', 'dob', 'description'], verbose=True)


if input_text:
    st.write(parent_chain({'name': input_text}))

    with st.expander("Person Name"):
        st.write(person_memory.buffer)

    with st.expander("Major Events"):
        st.write(description_memory.buffer)