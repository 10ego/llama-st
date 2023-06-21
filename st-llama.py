import streamlit as st

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
#from langchain.callbacks.streamlit import StreamlitCallbackHandler
from callbacks.streamlit import StreamlitCallbackHandler

# Streamlit Setup
st.title("Demo LLaMA")
st.write("Testing the 4bit quantized version of the Vicuna 7B 1.1 model")
input_container = st.container()
output_container = st.container()

# Chain Setup

model_path="/home/tkim/project/llama.cpp/models/vicuna-7b-1.1.ggmlv3.q4_0.bin"
template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])
callback_manager = CallbackManager([StreamlitCallbackHandler(output_container)])
llm = LlamaCpp(
    model_path=model_path, callback_manager=callback_manager, verbose=True
)
chain = LLMChain(prompt=prompt, llm=llm)

# Render

output_container.markdown("### Response:")

with input_container:
	user_input = input_container.text_input("", "", key="input")

with output_container:
	if user_input:
		response = chain.run(question=user_input) 
		output_container.write(response)
