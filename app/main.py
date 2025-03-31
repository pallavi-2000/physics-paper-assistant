import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOpenAI
from prompts import WAVE_PROMPT

# Load .env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Set up LLM
llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    temperature=0.7
)

st.set_page_config(page_title="Physics Paper Assistant", page_icon="🧠")
st.title("🧠 Physics Paper Assistant + Simulator")

st.markdown("Enter a concept below (e.g. *1D wave equation*) and get Julia code to simulate it.")

st.divider()
st.header("🔢 Simulation Code Generator")

sim_input = st.text_input("📝 Describe a physics system:", "1D wave equation with fixed boundaries")

if st.button("Generate Simulation Code"):
    with st.spinner("Thinking..."):
        prompt = WAVE_PROMPT.format(input=sim_input)
        response = llm.invoke([HumanMessage(content=prompt)])
        code = response.content

    st.code(code, language="julia")
    st.download_button("⬇️ Download code", data=code, file_name="wave_sim.jl", mime="text/plain")
