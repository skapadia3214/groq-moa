import copy
import json
from typing import Iterable, Dict, Any
import os
from dotenv import load_dotenv
import time
from functools import lru_cache

import streamlit as st
from streamlit_ace import st_ace
from groq import Groq
import tiktoken
from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header

from moa.agent import MOAgent
from moa.agent.moa import ResponseChunk, MOAgentConfig, get_available_models
from moa.agent.prompts import SYSTEM_PROMPT, REFERENCE_SYSTEM_PROMPT

# Set page config at the very beginning
st.set_page_config(
    page_title="Mixture-Of-Agents Powered by Groq",
    page_icon='static/favicon.ico',
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add this CSS block
st.markdown("""
<style>
    :root {
        --background-color: #ffffff;
        --text-color: #000000;
        --card-background: #f0f0f0;
    }
    .dark {
        --background-color: #1e1e1e;
        --text-color: #ffffff;
        --card-background: #2d2d2d;
    }
    body {
        color: var(--text-color);
        background-color: var(--background-color);
    }
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea textarea {
        background-color: var(--card-background);
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# Initialize theme
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Apply theme
st.markdown(f"<style>:root {{ --theme: {st.session_state.theme}; }}</style>", unsafe_allow_html=True)

# Specify the full path to your .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY. Please set it in your .env file or environment variables.")
    st.stop()

# Default configuration
default_main_agent_config = {
    "main_model": "llama-3.1-70b-versatile",
    "cycles": 3,
    "temperature": 0.1,
    "system_prompt": "You are a highly capable AI assistant with expertise in various fields. Your role is to provide helpful, accurate, and thoughtful responses to user queries. \n\nConsider the following points when formulating your response:\n1. Analyze the user's question carefully to understand the core of their inquiry.\n2. Draw upon your broad knowledge base to provide comprehensive answers.\n3. If applicable, consider multiple perspectives or approaches to the topic.\n4. Provide clear explanations and, when appropriate, examples to illustrate your points.\n5. If the query involves sensitive topics, maintain an objective and respectful tone.\n6. If you're unsure about any aspect of the answer, acknowledge the limitations of your knowledge.\n\n{helper_response}\n\nBased on the above considerations and any additional context provided, please respond to the user's query.\n",
    "reference_system_prompt": "As an advanced AI assistant, your task is to synthesize and refine responses from multiple AI models into a single, high-quality answer. Follow these guidelines:\n\n1. Critically evaluate the information provided in the responses, recognizing potential biases or inaccuracies.\n2. Identify common themes and key points across the responses.\n3. Assess the relevance and accuracy of each piece of information.\n4. Combine the most valuable insights from each response.\n5. Provide a well-structured, coherent, and comprehensive reply that goes beyond simply replicating the given answers.\n6. Ensure your response adheres to the highest standards of accuracy and reliability.\n7. If there are conflicting viewpoints, present them objectively and, if possible, explain the reasons for the differences.\n8. If appropriate, add any relevant information that may have been missed by the other models.\n\nResponses from models:\n{responses}\n\nUsing the above information and guidelines, craft a refined and authoritative response to the user's query.\n"
}

default_layer_agent_config = {
    "layer_agent_1": {
        "system_prompt": "Think through your response step by step. {helper_response}",
        "model_name": "llama-3.1-8b-instant",
        "temperature": 0.3
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "gemma2-9b-it",
        "temperature": 0.7
    },
    "layer_agent_3": {
        "system_prompt": "You are an expert at logic and reasoning. Always take a logical approach to the answer. {helper_response}",
        "model_name": "mixtral-8x7b-32768",
        "temperature": 0.1
    }
}

# Recommended configuration
rec_main_agent_config = {
    "main_model": "llama-3.1-70b-versatile",
    "cycles": 2,
    "temperature": 0.1,
    "system_prompt": "You are a personal assistant that is helpful.\n\n{helper_response}",
    "reference_system_prompt": "You have been provided with a set of responses from various open-source models to the latest user query. \nYour task is to synthesize these responses into a single, high-quality response. \nIt is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. \nYour response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. \nEnsure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\nResponses from models:\n{responses}\n"
}

rec_layer_agent_config = {
    "layer_agent_1": {
        "system_prompt": "Think through your response step by step. {helper_response}",
        "model_name": "llama-3.1-8b-instant",
        "temperature": 0.1
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "gemma2-9b-it",
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
    }
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
            if layer_outputs:
                st.subheader("Layer Agent Outputs", divider='rainbow')
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

def get_model_index(model_name, available_models):
    try:
        return available_models.index(model_name)
    except ValueError:
        return 0  # Default to the first model if not found

def initialize_config():
    if "moa_main_agent_config" not in st.session_state:
        st.session_state.moa_main_agent_config = copy.deepcopy(default_main_agent_config)
    if "moa_layer_agent_config" not in st.session_state:
        st.session_state.moa_layer_agent_config = copy.deepcopy(default_layer_agent_config)

def set_moa_agent(override: bool = False):
    initialize_config()
    if override or ("moa_agent" not in st.session_state):
        # Initialize Groq client
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        st.session_state.moa_agent = MOAgent.from_config(
            **st.session_state.moa_main_agent_config,
            layer_agent_config=st.session_state.moa_layer_agent_config,
            groq_client=groq_client
        )

def get_fallback_models():
    return [
        "llama-3.1-70b-chat",
        "mixtral-8x7b-32768",
        "llama-3.1-8b-instant",
        "gemma-7b-it"
    ]

@lru_cache(maxsize=None)
def get_cached_models():
    return get_available_models()

def get_available_models():
    return ['gemma2-9b-it','mixtral-8x7b-32768','llama-3.1-8b-instant','llama-3.1-70b-versatile']

def is_model_available(model_name):
    try:
        Groq(api_key=GROQ_API_KEY).chat.completions.create(model=model_name, messages=[{"role": "user", "content": "test"}])
        return True
    except Exception:
        return False

def export_chat_history():
    return "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages])

def save_configuration(config_name):
    config = {
        "main_agent_config": st.session_state.moa_main_agent_config,
        "layer_agent_config": st.session_state.moa_layer_agent_config
    }
    config_dir = os.path.join(os.path.dirname(__file__), "saved_configs")
    os.makedirs(config_dir, exist_ok=True)
    file_path = os.path.join(config_dir, f"{config_name}.json")
    with open(file_path, "w") as f:
        json.dump(config, f, indent=2)
    st.success(f"Configuration '{config_name}' saved successfully!")

def load_saved_configs():
    config_dir = os.path.join(os.path.dirname(__file__), "saved_configs")
    if not os.path.exists(config_dir):
        return []
    return [f.replace(".json", "") for f in os.listdir(config_dir) if f.endswith(".json")]

def load_configuration(config_name):
    config_dir = os.path.join(os.path.dirname(__file__), "saved_configs")
    file_path = os.path.join(config_dir, f"{config_name}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            config = json.load(f)
        st.session_state.moa_main_agent_config = config["main_agent_config"]
        st.session_state.moa_layer_agent_config = config["layer_agent_config"]
        set_moa_agent(override=True)
        st.success(f"Configuration '{config_name}' loaded successfully!")
        st.rerun()
    else:
        st.error(f"Configuration '{config_name}' not found!")

# Initialize saved configs in session state
if "saved_configs" not in st.session_state:
    st.session_state.saved_configs = {}

def estimate_token_usage(text: str) -> int:
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoder.encode(text))

def estimate_total_tokens(messages, layer_outputs):
    total_tokens = 0
    for message in messages:
        total_tokens += estimate_token_usage(message['content'])
    
    for layer, agents in layer_outputs.items():
        for agent, outputs in agents.items():
            total_tokens += estimate_token_usage("".join(outputs))
    
    return total_tokens

def estimate_cost(tokens: int, model: str) -> float:
    # These are placeholder values. Replace with actual pricing for your models.
    cost_per_1k_tokens = {
        "llama3-70b-8192": 0.0005,
        "gemini-1.5-pro-latest": 0.0001,
        # Add other models and their costs here
    }
    return (tokens / 1000) * cost_per_1k_tokens.get(model, 0.0001)  # Default to $0.0001 per 1k tokens if model not found

# Add this function to toggle the theme
def toggle_theme():
    if st.session_state.theme == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"
    st.rerun()

def update_main_model():
    new_model = st.session_state.main_model
    st.session_state.moa_main_agent_config["main_model"] = new_model
    set_moa_agent(override=True)

def update_layer_model(agent_key):
    new_model = st.session_state[f"{agent_key}_model"]
    st.session_state.moa_layer_agent_config[agent_key]["model_name"] = new_model
    set_moa_agent(override=True)

valid_model_names = get_cached_models()

st.markdown("<a href='https://groq.com'><img src='app/static/banner.png' width='500' style='display: block; margin-left: auto; margin-right: auto;'></a>", unsafe_allow_html=True)
st.divider()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize configs
initialize_config()
set_moa_agent()

# Sidebar navigation
with st.sidebar:
    st.title("MOA Configuration")
    
    # Add theme toggle
   # theme_toggle = st.toggle("Dark mode", value=st.session_state.theme == "dark")
    #if theme_toggle != (st.session_state.theme == "dark"):
     #   toggle_theme()
    
    selected = option_menu(
        menu_title=None,
        options=["Chat", "Settings", "Import/Export"],
        icons=["chat", "gear", "arrow-left-right"],
        menu_icon="cast",
        default_index=0,
    )

# Main content area
st.markdown("<h1 style='text-align: center; color: var(--primary-color);'>Mixture of Agents</h1>", unsafe_allow_html=True)

if selected == "Chat":
    st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .user-message {
        background-color: #E9F5E9;
        border-bottom-right-radius: 0;
    }
    .assistant-message {
        background-color: #F0F4F8;
        border-bottom-left-radius: 0;
    }
    .message-content {
        word-wrap: break-word;
    }
    .message-metadata {
        font-size: 0.8em;
        color: #888;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.header("MOA Chat", divider="rainbow")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"""
            <div class="chat-message {message['role']}-message">
                <div class="message-content">{message['content']}</div>
                <div class="message-metadata">
                    {message.get('timestamp', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if message['role'] == 'assistant' and 'layer_outputs' in message:
                with st.expander("Layer Agents' Responses", expanded=False):
                    for layer, agents in message['layer_outputs'].items():
                        st.markdown(f"**Layer {layer}**")
                        for agent, output in agents.items():
                            st.info(f"*{agent}:* {''.join(output)}")

    # User input
    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": time.strftime("%H:%M")})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            layer_outputs = {}

            # Simulate stream of response with milliseconds delay
            moa_agent: MOAgent = st.session_state.moa_agent
            conversation_context = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in st.session_state.messages[-10:]
            ])
            full_query = f"{conversation_context}\n\nUser: {prompt}\nAssistant:"
            
            for chunk in moa_agent.chat(full_query, output_format='json'):
                if chunk['response_type'] == 'intermediate':
                    layer = chunk['metadata']['layer']
                    agent = chunk['metadata'].get('agent', 'Unknown')
                    if layer not in layer_outputs:
                        layer_outputs[layer] = {}
                    if agent not in layer_outputs[layer]:
                        layer_outputs[layer][agent] = []
                    layer_outputs[layer][agent].append(chunk['delta'])
                else:
                    full_response += chunk['delta']
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # Add assistant message to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "layer_outputs": layer_outputs,
            "timestamp": time.strftime("%H:%M")
        })

        # Display layer outputs
        if layer_outputs:
            with st.expander("Layer Agents' Responses", expanded=False):
                for layer, agents in layer_outputs.items():
                    st.markdown(f"**Layer {layer}**")
                    for agent, output in agents.items():
                        st.info(f"*{agent}:* {''.join(output)}")

        # Estimate token usage and cost
        total_tokens = estimate_total_tokens(st.session_state.messages, layer_outputs)
        cost_estimate = estimate_cost(total_tokens, st.session_state.moa_main_agent_config['main_model'])
        
        st.session_state.usage_stats = {
            "total_tokens": total_tokens,
            "cost_estimate": cost_estimate
        }

    # Usage statistics
    if 'usage_stats' in st.session_state:
        st.sidebar.markdown("### Usage Statistics")
        st.sidebar.markdown(f"""
        - **Estimated tokens:** {st.session_state.usage_stats["total_tokens"]}
        - **Estimated cost:** ${st.session_state.usage_stats['cost_estimate']:.6f}
        """)

    # Chat controls
    st.sidebar.markdown("### Chat Controls")
    if st.sidebar.button("Clear Chat History", type="primary"):
        st.session_state.messages = []
        if 'usage_stats' in st.session_state:
            del st.session_state.usage_stats
        set_moa_agent(override=True)
        st.rerun()

    # Configuration expander
    with st.sidebar.expander("Current MOA Configuration", expanded=False):
        st.json(st.session_state.moa_main_agent_config)
        st.json(st.session_state.moa_layer_agent_config)

    # Add download button for chat history
    if st.session_state.messages:
        chat_history = export_chat_history()
        st.download_button(
            label="Download Chat History",
            data=chat_history,
            file_name="moa_chat_history.txt",
            mime="text/plain"
        )

elif selected == "Settings":
    st.header("MOA Settings", divider="rainbow")
    
    # Add buttons to load configurations
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Default Configuration"):
            st.session_state.moa_main_agent_config = copy.deepcopy(default_main_agent_config)
            st.session_state.moa_layer_agent_config = copy.deepcopy(default_layer_agent_config)
            set_moa_agent(override=True)
            st.success("Default configuration loaded successfully!")
            st.rerun()
    with col2:
        if st.button("Load Recommended Configuration"):
            st.session_state.moa_main_agent_config = copy.deepcopy(rec_main_agent_config)
            st.session_state.moa_layer_agent_config = copy.deepcopy(rec_layer_agent_config)
            set_moa_agent(override=True)
            st.success("Recommended configuration loaded successfully!")
            st.rerun()

    # Main Agent Configuration
    st.subheader("Main Agent Configuration")
    
    # Model selection for main agent
    main_model_index = get_model_index(st.session_state.moa_main_agent_config["main_model"], valid_model_names)
    new_main_model = st.selectbox(
        "Select Main Model",
        options=valid_model_names,
        index=main_model_index,
        key="main_model"
    )
    if new_main_model != st.session_state.moa_main_agent_config["main_model"]:
        update_main_model()

    # Other main agent settings
    new_cycles = st.number_input("Cycles", min_value=1, max_value=10, value=st.session_state.moa_main_agent_config["cycles"])
    if new_cycles != st.session_state.moa_main_agent_config["cycles"]:
        st.session_state.moa_main_agent_config["cycles"] = new_cycles
        set_moa_agent(override=True)
        st.success(f"Cycles updated to {new_cycles}")
        st.rerun()

    new_temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=st.session_state.moa_main_agent_config["temperature"], step=0.1)
    if new_temperature != st.session_state.moa_main_agent_config["temperature"]:
        st.session_state.moa_main_agent_config["temperature"] = new_temperature
        set_moa_agent(override=True)
        st.success(f"Temperature updated to {new_temperature}")
        st.rerun()
    
    # Layer Agent Configuration
    st.subheader("Layer Agent Configuration")
    
    # Number of Layers
    st.subheader("Layer Configuration")
    num_layers = st.number_input("Number of Layers", min_value=1, max_value=5, value=len(st.session_state.moa_layer_agent_config), step=1)
    
    if num_layers != len(st.session_state.moa_layer_agent_config):
        new_layer_config = {}
        for i in range(1, num_layers + 1):
            layer_key = f"layer_agent_{i}"
            if layer_key in st.session_state.moa_layer_agent_config:
                new_layer_config[layer_key] = st.session_state.moa_layer_agent_config[layer_key]
            else:
                new_layer_config[layer_key] = {
                    "system_prompt": f"You are layer agent {i}. Provide your unique perspective. {{helper_response}}",
                    "model_name": valid_model_names[0],
                    "temperature": 0.5
                }
        st.session_state.moa_layer_agent_config = new_layer_config
        set_moa_agent(override=True)
        st.success(f"Number of layers updated to {num_layers}")
        st.rerun()
    
    for agent_key, agent_config in st.session_state.moa_layer_agent_config.items():
        with st.expander(f"{agent_key} Configuration"):
            # Model selection for layer agent
            model_index = get_model_index(agent_config["model_name"], valid_model_names)
            new_layer_model = st.selectbox(
                f"Select Model for {agent_key}",
                options=valid_model_names,
                index=model_index,
                key=f"{agent_key}_model"
            )
            if new_layer_model != agent_config["model_name"]:
                update_layer_model(agent_key)
            
            # Other layer agent settings
            new_layer_temperature = st.slider(f"Temperature for {agent_key}", min_value=0.0, max_value=1.0, value=agent_config["temperature"], step=0.1, key=f"{agent_key}_temperature")
            if new_layer_temperature != agent_config["temperature"]:
                agent_config["temperature"] = new_layer_temperature
                set_moa_agent(override=True)
                st.success(f"Temperature for {agent_key} updated to {new_layer_temperature}")
                st.rerun()

            new_system_prompt = st.text_area(f"System Prompt for {agent_key}", value=agent_config["system_prompt"], key=f"{agent_key}_system_prompt")
            if new_system_prompt != agent_config["system_prompt"]:
                agent_config["system_prompt"] = new_system_prompt
                set_moa_agent(override=True)
                st.success(f"System prompt for {agent_key} updated")
                st.rerun()

    # Configuration expander
    with st.expander("Current MOA Configuration", expanded=False):
        st.json(st.session_state.moa_main_agent_config)
        st.json(st.session_state.moa_layer_agent_config)

elif selected == "Import/Export":
    st.header("Import/Export Configuration", divider="rainbow")
    
    # Export configuration
    st.subheader("Export Configuration")
    config_name = st.text_input("Enter a name for the configuration")
    if st.button("Export"):
        if config_name:
            save_configuration(config_name)
        else:
            st.warning("Please enter a name for the configuration.")
    
    # Import configuration
    st.subheader("Import Configuration")
    saved_configs = load_saved_configs()
    if saved_configs:
        selected_config = st.selectbox("Select a saved configuration", saved_configs)
        if st.button("Import"):
            load_configuration(selected_config)
    else:
        st.info("No saved configurations found.")

# At the end of your app
st.markdown(f'</div>', unsafe_allow_html=True)
