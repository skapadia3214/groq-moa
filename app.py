import copy
import json
from typing import Iterable, Dict, Any

import streamlit as st
from streamlit_ace import st_ace
from groq import Groq

from moa.agent import MOAgent
from moa.agent.moa import ResponseChunk, MOAgentConfig
from moa.agent.prompts import SYSTEM_PROMPT, REFERENCE_SYSTEM_PROMPT

# Default configuration
default_main_agent_config = {
    "main_model": "llama3-70b-8192",
    "cycles": 3,
    "temperature": 0.1,
    "system_prompt": SYSTEM_PROMPT,
    "reference_system_prompt": REFERENCE_SYSTEM_PROMPT
}

default_layer_agent_config = {
    "layer_agent_1": {
        "system_prompt": "Think through your response step by step. {helper_response}",
        "model_name": "llama3-8b-8192",
        "temperature": 0.3
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "gemma-7b-it",
        "temperature": 0.7
    },
    "layer_agent_3": {
        "system_prompt": "You are an expert at logic and reasoning. Always take a logical approach to the answer. {helper_response}",
        "model_name": "llama3-8b-8192",
        "temperature": 0.1
    },
}

# Recommended Configuration
rec_main_agent_config = {
    "main_model": "llama-3.1-70b-versatile",
    "cycles": 2,
    "temperature": 0.1,
    "system_prompt": SYSTEM_PROMPT,
    "reference_system_prompt": REFERENCE_SYSTEM_PROMPT
}

rec_layer_agent_config = {
    "layer_agent_1": {
        "system_prompt": "Think through your response step by step. {helper_response}",
        "model_name": "gemma2-9b-it",
        "temperature": 0.1
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "llama-3.1-8b-instant",
        "temperature": 0.2,
        "max_tokens": 2048
    },
    "layer_agent_3": {
        "system_prompt": "You are an expert at logic and reasoning. Always take a logical approach to the answer. {helper_response}",
        "model_name": "llama-3.1-70b-versatile",
        "temperature": 0.4,
        "max_tokens": 2048
    },
    "layer_agent_4": {
        "system_prompt": "You are an expert planner agent. Create a plan for how to answer the human's query. {helper_response}",
        "model_name": "mixtral-8x7b-32768",
        "temperature": 0.5
    },
}

# Helper functions
def json_to_moa_config(config_file) -> Dict[str, Any]:
    config = json.load(config_file)
    moa_config = MOAgentConfig( # To check if everything is ok
        **config
    ).model_dump(exclude_unset=True)
    return {
        'moa_layer_agent_config':moa_config.pop('layer_agent_config', None),
        'moa_main_agent_config': moa_config or None
    }

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

def set_moa_agent(
    moa_main_agent_config = None,
    moa_layer_agent_config = None,
    override: bool = False
):
    moa_main_agent_config = copy.deepcopy(moa_main_agent_config or default_main_agent_config)
    moa_layer_agent_config = copy.deepcopy(moa_layer_agent_config or default_layer_agent_config)

    if "moa_main_agent_config" not in st.session_state or override:
        st.session_state.moa_main_agent_config = moa_main_agent_config

    if "moa_layer_agent_config" not in st.session_state or override:
        st.session_state.moa_layer_agent_config = moa_layer_agent_config

    if override or ("moa_agent" not in st.session_state):
        st_main_copy = copy.deepcopy(st.session_state.moa_main_agent_config)
        st_layer_copy = copy.deepcopy(st.session_state.moa_layer_agent_config)
        st.session_state.moa_agent = MOAgent.from_config(
            **st_main_copy,
            layer_agent_config=st_layer_copy
        )

        del st_main_copy
        del st_layer_copy

    del moa_main_agent_config
    del moa_layer_agent_config

# App
st.set_page_config(
    page_title="Mixture-Of-Agents Powered by Groq",
    page_icon='static/favicon.ico',
        menu_items={
        'About': "## Groq Mixture-Of-Agents \n Powered by [Groq](https://groq.com)"
    },
    layout="wide"
)

valid_model_names = [model.id for model in Groq().models.list().data if not (model.id.startswith("whisper") or model.id.startswith("llama-guard"))]

st.markdown("<a href='https://groq.com'><img src='app/static/banner.png' width='500'></a>", unsafe_allow_html=True)
st.write("---")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

set_moa_agent()

# Sidebar for configuration
with st.sidebar:
    st.title("MOA Configuration")
    # upl_col, load_col = st.columns(2)
    st.download_button(
        "Download Current MoA Configuration as JSON", 
        data=json.dumps({
            **st.session_state.moa_main_agent_config,
            'moa_layer_agent_config': st.session_state.moa_layer_agent_config
        }, indent=2),
        file_name="moa_config.json"
    )

    # moa_config_upload = st.file_uploader("Choose a JSON file", type="json")
    # submit_config_file = st.button("Update config")
    # if moa_config_upload and submit_config_file:
    #     try:
    #         moa_config = json_to_moa_config(moa_config_upload)
    #         set_moa_agent(
    #             moa_main_agent_config=moa_config['moa_main_agent_config'],
    #             moa_layer_agent_config=moa_config['moa_layer_agent_config']
    #         )
    #         st.session_state.messages = []
    #         st.success("Configuration updated successfully!")
    #     except Exception as e:
    #         st.error(f"Error loading file: {str(e)}")

    with st.form("Agent Configuration", border=False):
        # Load and Save moa config file
             
        if st.form_submit_button("Use Recommended Config"):
            try:
                set_moa_agent(
                    moa_main_agent_config=rec_main_agent_config,
                    moa_layer_agent_config=rec_layer_agent_config,
                    override=True
                )
                st.session_state.messages = []
                st.success("Configuration updated successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON in Layer Agent Configuration. Please check your input.")
            except Exception as e:
                st.error(f"Error updating configuration: {str(e)}")

        # Main model selection
        new_main_model = st.selectbox(
            "Select Main Model",
            options=valid_model_names,
            index=valid_model_names.index(st.session_state.moa_main_agent_config['main_model'])
        )



        # Cycles input
        new_cycles = st.number_input(
            "Number of Layers",
            min_value=1,
            max_value=10,
            value=st.session_state.moa_main_agent_config['cycles']
        )

        # Main Model Temperature
        main_temperature = st.number_input(
            label="Main Model Temperature",
            value=0.1,
            min_value=0.0,
            max_value=1.0,
            step=0.1
        )

        # Layer agent configuration
        tooltip = "Agents in the layer agent configuration run in parallel _per cycle_. Each layer agent supports all initialization parameters of [Langchain's ChatGroq](https://api.python.langchain.com/en/latest/chat_models/langchain_groq.chat_models.ChatGroq.html) class as valid dictionary fields."
        st.markdown("Layer Agent Config", help=tooltip)
        new_layer_agent_config = st_ace(
            key="layer_agent_config",
            value=json.dumps(st.session_state.moa_layer_agent_config, indent=2),
            language='json',
            placeholder="Layer Agent Configuration (JSON)",
            show_gutter=False,
            wrap=True,
            auto_update=True
        )

        with st.expander("Optional Main Agent Params"):
            tooltip_str = """\
Main Agent configuration that will respond to the user based on the layer agent outputs.
Valid fields:
- ``system_prompt``: System prompt given to the main agent. \
**IMPORTANT**: it should always include a `{helper_response}` prompt variable.
- ``reference_prompt``: This prompt is used to concatenate and format each layer agent's output into one string. \
This is passed into the `{helper_response}` variable in the system prompt. \
**IMPORTANT**: it should always include a `{responses}` prompt variable. 
- ``main_model``: Which Groq powered model to use. Will overwrite the model given in the dropdown.\
"""
            tooltip = tooltip_str
            st.markdown("Main Agent Config", help=tooltip)
            new_main_agent_config = st_ace(
                key="main_agent_params",
                value=json.dumps(st.session_state.moa_main_agent_config, indent=2),
                language='json',
                placeholder="Main Agent Configuration (JSON)",
                show_gutter=False,
                wrap=True,
                auto_update=True
            )

        if st.form_submit_button("Update Configuration"):
            try:
                new_layer_config = json.loads(new_layer_agent_config)
                new_main_config = json.loads(new_main_agent_config)
                # Configure conflicting params
                # If param in optional dropdown == default param set, prefer using explicitly defined param
                if new_main_config.get('main_model', default_main_agent_config['main_model']) == default_main_agent_config["main_model"]:
                    new_main_config['main_model'] = new_main_model
                
                if new_main_config.get('cycles', default_main_agent_config['cycles']) == default_main_agent_config["cycles"]:
                    new_main_config['cycles'] = new_cycles

                if new_main_config.get('temperature', default_main_agent_config['temperature']) == default_main_agent_config['temperature']:
                    new_main_config['temperature'] = main_temperature
                
                set_moa_agent(
                    moa_main_agent_config=new_main_config,
                    moa_layer_agent_config=new_layer_config,
                    override=True
                )
                st.session_state.messages = []
                st.success("Configuration updated successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON in Layer Agent Configuration. Please check your input.")
            except Exception as e:
                st.error(f"Error updating configuration: {str(e)}")

    st.markdown("---")
    st.markdown("""
    ### Credits
    - MOA: [Together AI](https://www.together.ai/blog/together-moa)
    - LLMs: [Groq](https://groq.com/)
    - Paper: [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)
    """)

# Main app layout
st.header("Mixture of Agents", anchor=False)
st.write("A demo of the Mixture of Agents architecture proposed by Together AI, Powered by Groq LLMs.")

# Display current configuration
with st.status("Current MOA Configuration", expanded=True, state='complete') as config_status:
    st.image("./static/moa_groq.svg", caption="Mixture of Agents Workflow", use_column_width='always')
    st.markdown(f"**Main Agent Config**:")
    new_layer_agent_config = st_ace(
        value=json.dumps(st.session_state.moa_main_agent_config, indent=2),
        language='json',
        placeholder="Layer Agent Configuration (JSON)",
        show_gutter=False,
        wrap=True,
        readonly=True,
        auto_update=True
    )
    st.markdown(f"**Layer Agents Config**:")
    new_layer_agent_config = st_ace(
        value=json.dumps(st.session_state.moa_layer_agent_config, indent=2),
        language='json',
        placeholder="Layer Agent Configuration (JSON)",
        show_gutter=False,
        wrap=True,
        readonly=True,
        auto_update=True
    )

if st.session_state.get("message", []) != []:
    st.session_state['expand_config'] = False
# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question"):
    config_status.update(expanded=False)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    moa_agent: MOAgent = st.session_state.moa_agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        ast_mess = stream_response(moa_agent.chat(query, output_format='json'))
        response = st.write_stream(ast_mess)
    
    st.session_state.messages.append({"role": "assistant", "content": response})