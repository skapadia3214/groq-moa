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



class MOAgentConfig(BaseModel):
    main_model: Optional[str] = None
    system_prompt: Optional[str] = None
    cycles: int = Field(...)
    layer_agent_config: Optional[Dict[str, Any]] = None
    reference_system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None

    class Config:
        extra = "allow"  # This allows for additional fields not explicitly defined

load_dotenv()
valid_model_names = Literal[
    'llama3-70b-8192',
    'llama3-8b-8192',
    'gemma-7b-it',
    'gemma2-9b-it',
    'mixtral-8x7b-32768',
    'llama-3.1-8b-instant',
    'llama-3.1-70b-versatile'
]

class ResponseChunk(TypedDict):
    delta: str
    response_type: Literal['intermediate', 'output']
    metadata: Dict[str, Any]


class MOAgent:
    def __init__(
        self,
        main_agent: RunnableSerializable[Dict, str],
        layer_agent: RunnableSerializable[Dict, Dict],
        reference_system_prompt: Optional[str] = None,
        cycles: Optional[int] = None,
        chat_memory: Optional[ConversationBufferMemory] = None
    ) -> None:
        self.reference_system_prompt = reference_system_prompt or REFERENCE_SYSTEM_PROMPT
        self.main_agent = main_agent
        self.layer_agent = layer_agent
        self.cycles = cycles or 1
        self.chat_memory = chat_memory or ConversationBufferMemory(
            memory_key="messages",
            return_messages=True
        )

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
    def from_config(
        cls,
        main_model: Optional[valid_model_names] = 'llama3-70b-8192',
        system_prompt: Optional[str] = None,
        cycles: int = 1,
        layer_agent_config: Optional[Dict] = None,
        reference_system_prompt: Optional[str] = None,
        **main_model_kwargs
    ):
        reference_system_prompt = reference_system_prompt or REFERENCE_SYSTEM_PROMPT
        system_prompt = system_prompt or SYSTEM_PROMPT
        layer_agent = MOAgent._configure_layer_agent(layer_agent_config)
        main_agent = MOAgent._create_agent_from_system_prompt(
            system_prompt=system_prompt,
            model_name=main_model,
            **main_model_kwargs
        )
        return cls(
            main_agent=main_agent,
            layer_agent=layer_agent,
            reference_system_prompt=reference_system_prompt,
            cycles=cycles
        )

    @staticmethod
    def _configure_layer_agent(
        layer_agent_config: Optional[Dict] = None
    ) -> RunnableSerializable[Dict, Dict]:
        if not layer_agent_config:
            layer_agent_config = {
                'layer_agent_1' : {'system_prompt': SYSTEM_PROMPT, 'model_name': 'llama3-8b-8192'},
                'layer_agent_2' : {'system_prompt': SYSTEM_PROMPT, 'model_name': 'gemma-7b-it'},
                'layer_agent_3' : {'system_prompt': SYSTEM_PROMPT, 'model_name': 'mixtral-8x7b-32768'}
            }

        parallel_chain_map = dict()
        for key, value in layer_agent_config.items():
            chain = MOAgent._create_agent_from_system_prompt(
                system_prompt=value.pop("system_prompt", SYSTEM_PROMPT), 
                model_name=value.pop("model_name", 'llama3-8b-8192'),
                **value
            )
            parallel_chain_map[key] = RunnablePassthrough() | chain
        
        chain = parallel_chain_map | RunnableLambda(MOAgent.concat_response)
        return chain

    @staticmethod
    def _create_agent_from_system_prompt(
        system_prompt: str = SYSTEM_PROMPT,
        model_name: str = "llama3-8b-8192",
        **llm_kwargs
    ) -> RunnableSerializable[Dict, str]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages", optional=True),
            ("human", "{input}")
        ])

        assert 'helper_response' in prompt.input_variables
        llm = ChatGroq(model=model_name, **llm_kwargs)
        
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
        llm_inp = {
            'input': input,
            'messages': messages or self.chat_memory.load_memory_variables({})['messages'],
            'helper_response': ""
        }
        for cyc in range(cycles):
            layer_output = self.layer_agent.invoke(llm_inp)
            l_frm_resp = layer_output['formatted_response']
            l_resps = layer_output['responses']
            
            llm_inp = {
                'input': input,
                'messages': self.chat_memory.load_memory_variables({})['messages'],
                'helper_response': l_frm_resp
            }

            if output_format == 'json':
                for l_out in l_resps:
                    yield ResponseChunk(
                        delta=l_out,
                        response_type='intermediate',
                        metadata={'layer': cyc + 1}
                    )

        stream = self.main_agent.stream(llm_inp)
        response = ""
        for chunk in stream:
            if output_format == 'json':
                    yield ResponseChunk(
                        delta=chunk,
                        response_type='output',
                        metadata={}
                    )
            else:
                yield chunk
            response += chunk

        if save:
            self.chat_memory.save_context({'input': input}, {'output': response})