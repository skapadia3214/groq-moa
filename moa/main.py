from agent import MOAgent

# Configure agent
layer_agent_config = {
    'layer_agent_1' : {'system_prompt': "Think through your response with step by step {helper_response}", 'model_name': 'llama3-8b-8192'},
    'layer_agent_2' : {'system_prompt': "Respond with a thought and then your response to the question {helper_response}", 'model_name': 'gemma-7b-it'},
    'layer_agent_3' : {'model_name': 'llama3-8b-8192'},
    'layer_agent_4' : {'model_name': 'gemma-7b-it'},
    'layer_agent_5' : {'model_name': 'llama3-8b-8192'},
}
agent = MOAgent.from_config(
    main_model='mixtral-8x7b-32768',
    layer_agent_config=layer_agent_config
)

while True:
    inp = input("\nAsk a question: ")
    stream = agent.chat(inp, output_format='json')
    for chunk in stream:
        print(chunk)