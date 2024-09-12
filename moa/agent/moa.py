"""
Langchain agent
"""
from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSerializable
from langchain_core.output_parsers import StrOutputParser

from .prompts import SYSTEM_PROMPT, REFERENCE_SYSTEM_PROMPT

from groq import Groq

# Update the valid_model_names
valid_model_names = Literal[
    'gemma2-9b-it',
    'mixtral-8x7b-32768',
    'llama-3.1-8b-instant',
    'llama-3.1-70b-versatile'
]

class MOAgentConfig(BaseModel):
    main_model: Optional[str] = None
    system_prompt: Optional[str] = None
    cycles: int = Field(...)
    layer_agent_config: Optional[Dict[str, Any]] = None
    reference_system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None

    class Config:
        extra = "allow"  # This allows for additional fields not explicitly defined

class ResponseChunk(TypedDict):
    delta: str
    response_type: Literal['intermediate', 'output']
    metadata: Dict[str, Any]

class MOAgent:
    def __init__(
        self,
        main_agent: RunnableSerializable[Dict, str],
        layer_agent: RunnableSerializable[Dict, Dict],
        layer_agent_config: Dict[str, Any],
        reference_system_prompt: Optional[str] = None,
        cycles: Optional[int] = None,
        chat_memory: Optional[ConversationBufferMemory] = None,
        groq_client: Optional[Groq] = None
    ) -> None:
        self.reference_system_prompt = reference_system_prompt or REFERENCE_SYSTEM_PROMPT
        self.main_agent = main_agent
        self.layer_agent = layer_agent
        self.layer_agent_config = layer_agent_config
        self.cycles = cycles or 1
        self.chat_memory = chat_memory or ConversationBufferMemory(
            memory_key="messages",
            return_messages=True
        )
        self.groq_client = groq_client

    @staticmethod
    def concat_response(
        inputs: Dict[str, str],
        reference_system_prompt: Optional[str] = None
    ):
        reference_system_prompt = reference_system_prompt or REFERENCE_SYSTEM_PROMPT

        responses = ""
        res_list = []
        for i, out in enumerate(inputs.values()):
            responses += f"{i}. {out}\n"
            res_list.append(out)

        formatted_prompt = reference_system_prompt.format(responses=responses)
        return {
            'formatted_response': formatted_prompt,
            'responses': res_list
        }

    @classmethod
    def from_config(cls, groq_client: Groq, **kwargs):
        config = MOAgentConfig(**kwargs)
        reference_system_prompt = config.reference_system_prompt or REFERENCE_SYSTEM_PROMPT
        system_prompt = config.system_prompt or SYSTEM_PROMPT
        layer_agent_config = config.layer_agent_config or {}
        layer_agent = MOAgent._configure_layer_agent(layer_agent_config, groq_client)
        main_agent = MOAgent._create_agent_from_system_prompt(
            system_prompt=system_prompt,
            model_name=config.main_model,
            groq_client=groq_client,
            **{k: v for k, v in kwargs.items() if k not in MOAgentConfig.__fields__}
        )
        return cls(
            main_agent=main_agent,
            layer_agent=layer_agent,
            layer_agent_config=layer_agent_config,
            reference_system_prompt=reference_system_prompt,
            cycles=config.cycles,
            groq_client=groq_client
        )

    @staticmethod
    def _configure_layer_agent(
        layer_agent_config: Optional[Dict] = None,
        groq_client: Optional[Groq] = None
    ) -> RunnableSerializable[Dict, Dict]:
        if not layer_agent_config:
            layer_agent_config = {
                'layer_agent_1' : {'system_prompt': SYSTEM_PROMPT, 'model_name': 'llama-3.1-8b-instant'},
                'layer_agent_2' : {'system_prompt': SYSTEM_PROMPT, 'model_name': 'gemma2-9b-it'},
                'layer_agent_3' : {'system_prompt': SYSTEM_PROMPT, 'model_name': 'mixtral-8x7b-32768'}
            }

        parallel_chain_map = dict()
        for key, value in layer_agent_config.items():
            model_name = value.get("model_name", 'Unknown Model')
            system_prompt = value.get("system_prompt", SYSTEM_PROMPT)
            chain = MOAgent._create_agent_from_system_prompt(
                system_prompt=system_prompt,
                model_name=model_name,
                groq_client=groq_client,
                **{k: v for k, v in value.items() if k not in ['system_prompt', 'model_name']}
            )
            parallel_chain_map[key] = RunnablePassthrough() | chain
        
        chain = parallel_chain_map | RunnableLambda(MOAgent.concat_response)
        return chain

    @staticmethod
    def _create_agent_from_system_prompt(
        system_prompt: str = SYSTEM_PROMPT,
        model_name: str = "llama-3.1-8b-instant",
        groq_client: Optional[Groq] = None,
        **llm_kwargs
    ) -> RunnableSerializable[Dict, str]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        # Remove 'type' from llm_kwargs if present
        llm_kwargs.pop('type', None)
        llm = ChatGroq(model=model_name, client=groq_client.chat.completions, **llm_kwargs)
        
        chain = prompt | llm | StrOutputParser()
        return chain

    def chat(
        self, 
        input: str,
        messages: Optional[List[BaseMessage]] = None,
        cycles: Optional[int] = None,
        save: bool = True,
        output_format: Literal['string', 'json'] = 'string'
    ) -> Generator[str | ResponseChunk, None, None]:
        cycles = cycles or self.cycles
        helper_response = ""  # Initialize helper_response
        for cyc in range(cycles):
            layer_outputs = []
            for agent_key, agent_config in self.layer_agent_config.items():
                model_name = agent_config.get('model_name', 'Unknown Model')
                system_prompt = agent_config.get('system_prompt', SYSTEM_PROMPT)
                temperature = agent_config.get('temperature', 0.7)

                llm_inp = {
                    'input': f"{input}\n\nPrevious layer outputs: {' '.join(layer_outputs)}",
                    'messages': [],  # We don't pass message history to Groq models
                    'helper_response': helper_response  # Add helper_response to the input
                }

                chain = self._create_agent_from_system_prompt(
                    system_prompt=system_prompt,
                    model_name=model_name,
                    groq_client=self.groq_client,
                    temperature=temperature
                )
                layer_output = chain.invoke(llm_inp)
                layer_outputs.append(layer_output)
                helper_response += f"{agent_key}: {layer_output}\n"  # Update helper_response

                if output_format == 'json':
                    yield ResponseChunk(
                        delta=layer_output,
                        response_type='intermediate',
                        metadata={'layer': cyc + 1, 'agent': f"{agent_key} ({model_name})"}
                    )

        # Main agent
        main_input = f"{input}\n\nLayer agent outputs: {helper_response}"
        stream = self.main_agent.stream({'input': main_input, 'messages': [], 'helper_response': helper_response})
        response = ""
        for chunk in stream:
            if output_format == 'json':
                yield ResponseChunk(
                    delta=chunk,
                    response_type='output',
                    metadata={'agent': f"Main Agent ({self.layer_agent_config.get('main_model', 'Unknown Model')})"}
                )
            else:
                yield chunk
            response += chunk

        if save:
            self.chat_memory.save_context({'input': input}, {'output': response})

def get_available_models():
    return [
        "gemma2-9b-it",
        "mixtral-8x7b-32768",
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile"
    ]

# Make sure to export the function
__all__ = ["MOAgent", "MOAgentConfig", "ResponseChunk", "get_available_models"]
