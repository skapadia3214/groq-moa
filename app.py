import streamlit as st
import json
from typing import Iterable
from moa.agent import MOAgent
from moa.agent.moa import ResponseChunk


valid_model_names = [
    'llama3-70b-8192',
    'llama3-8b-8192',
    'gemma-7b-it',
    'mixtral-8x7b-32768'
]

def stream_response(messages: Iterable[ResponseChunk]):
    layer_outputs = {}
    for message in messages:
        if message['response_type'] == 'intermediate':
            layer = message['metadata']['layer']
            if layer not in layer_outputs:
                layer_outputs[layer] = []
            layer_outputs[layer].append(message['delta'])
        else:
            # Display accumulated layer outputs
            for layer, outputs in layer_outputs.items():
                st.write(f"Layer {layer}")
                cols = st.columns(len(outputs))
                for i, output in enumerate(outputs):
                    with cols[i]:
                        st.expander(label=f"Agent {i+1}", expanded=False).write(output)
            
            # Clear layer outputs for the next iteration
            layer_outputs = {}
            
            # Yield the main agent's output
            yield message['delta']

# Default configuration
default_config = {
    "main_model": "llama3-70b-8192",
    "cycles": 1,
    "layer_agent_config": {
        "layer_agent_1": {
            "system_prompt": "Think through your response with step by step {helper_response}",
            "model_name": "llama3-8b-8192"
        },
        "layer_agent_2": {
            "system_prompt": "Respond with a thought and then your response to the question {helper_response}",
            "model_name": "gemma-7b-it"
        },
        "layer_agent_3": {"model_name": "llama3-8b-8192"},
        "layer_agent_4": {"model_name": "gemma-7b-it"}
    }
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "moa_config" not in st.session_state:
    st.session_state.moa_config = default_config

if "moa_agent" not in st.session_state:
    st.session_state.moa_agent = MOAgent.from_config(**st.session_state.moa_config)

# Sidebar for configuration
with st.sidebar:
    config_form = st.form("Agent Configuration", border=False)
    config_form.title("MOA Configuration")

    # Main model selection
    main_model = config_form.selectbox(
        "Select Main Model",
        options=valid_model_names,
        index=valid_model_names.index(st.session_state.moa_config["main_model"])
    )

    # Cycles input
    cycles = config_form.number_input(
        "Number of Cycles",
        min_value=1,
        max_value=10,
        value=st.session_state.moa_config["cycles"]
    )

    # Layer agent configuration
    layer_agent_config = config_form.text_area(
        "Layer Agent Configuration (JSON)",
        json.dumps(st.session_state.moa_config["layer_agent_config"], indent=2),
        height=500
    )

    if config_form.form_submit_button("Update Configuration"):
        try:
            new_layer_config = json.loads(layer_agent_config)
            new_config = {
                "main_model": main_model,
                "cycles": cycles,
                "layer_agent_config": new_layer_config
            }
            st.session_state.moa_config = new_config
            st.session_state.moa_agent = MOAgent.from_config(**new_config)
            st.session_state.messages = []
            st.success("Configuration updated successfully!")
        except json.JSONDecodeError:
            st.error("Invalid JSON in Layer Agent Configuration. Please check your input.")
        except Exception as e:
            st.error(f"Error updating configuration: {str(e)}")

# Main app layout
st.title("Mixture of Agents")
st.write("A demo of the Mixture of Agents architecture proposed by Together AI, Powered by Groq LLMs.")
st.image("./static/moa_arc.png", caption="Source: https://www.together.ai/blog/together-moa")

# Display current configuration
with st.expander("Current MOA Configuration", expanded=False):
    st.json(st.session_state.moa_config)

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    moa_agent: MOAgent = st.session_state.moa_agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        ast_mess = stream_response(moa_agent.chat(query, output_format='json'))
        response = st.write_stream(ast_mess)
    
    st.session_state.messages.append({"role": "assistant", "content": response})