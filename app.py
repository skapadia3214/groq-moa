import streamlit as st
from typing import Iterable

from moa.agent import MOAgent
from moa.agent.moa import ResponseChunk
from moa.agent.prompts import SYSTEM_PROMPT, REFERENCE_SYSTEM_PROMPT

LAYER_AGENT_CONFIG = {
    'layer_agent_1' : {'system_prompt': "Think through your response with step by step {helper_response}", 'model_name': 'llama3-8b-8192'},
    'layer_agent_2' : {'system_prompt': "Respond with a thought and then your response to the question {helper_response}", 'model_name': 'gemma-7b-it'},
    'layer_agent_3' : {'model_name': 'llama3-8b-8192'},
    'layer_agent_4' : {'model_name': 'gemma-7b-it'},
}
MAIN_MODEL = 'llama3-70b-8192'
CYCLES = 1
def stream_response(messages: Iterable[ResponseChunk]):
    layer_expander = st.expander(label=f"Layer")
    for message in messages:
        if message['response_type'] == 'intermediate':
            output_expander = st.expander(label=f"Layer Agent Output")
            with layer_expander:
                output_expander.write(message['delta'])
        else:
            yield message['delta']

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Mixture of Agents")
st.write("A demo of the Mixture of Agents architecture proposed by Together AI, Powered by Groq LLMs.")
st.image("./static/moa_arc.png", caption="Source: https://www.together.ai/blog/together-moa")

if "moa_agent" not in st.session_state:

    st.session_state['moa_agent'] = MOAgent.from_config(
        main_model='llama3-8b-8192',
        system_prompt=SYSTEM_PROMPT, # Make sure this contains a {helper_response} variable in the string
        cycles=1,
        layer_agent_config=LAYER_AGENT_CONFIG,
        reference_system_prompt=REFERENCE_SYSTEM_PROMPT # Make sure this contains a {responses} variable in the string
    )

with st.expander("MOA Configuration", icon=":material/settings:"):
    st.markdown(f"**Main Model**: ``{MAIN_MODEL}``")
    st.markdown(f"**Cycles**: ``{CYCLES}``")
    cols = st.columns(len(LAYER_AGENT_CONFIG))
    for i, (key, val) in enumerate(LAYER_AGENT_CONFIG.items()):
        format_vals = ""
        for param, v in val.items():
            format_vals += f"_{param}_: ``{v}``\n"
        cols[i].markdown(f"**{key}**\n{format_vals}")

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


if query := st.chat_input("Ask a Question"):
    st.session_state.messages.append({'role': 'user', 'content': query})
    with st.chat_message("user"):
        st.write(query)

    moa_agent: MOAgent = st.session_state['moa_agent']
    ast_mess = stream_response(moa_agent.chat(query, output_format='json'))
    response = st.write_stream(ast_mess)

    st.session_state.messages.append({
        'role': 'assistant',
        'content': response
    })