from config import logger
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain.chat_models import init_chat_model

from .classes import RespondToUser
from .constructor import PromptConstructor

class ChatAgent:
    """
    A chat agent that manages interactions with the user.
    
    This class handles the initialization, configuration, and interaction with different
    language models (LLMs) such as GPT, LLAMA, Mistral, etc. It manages prompt construction,
    response formatting, and tool integration.
    
    Attributes:
        model_selection (str): The identifier for the selected language model (e.g., "LLAMA", "GPT").
        base_url (str): The base URL for API connections, defaults to localhost for local models.
        language (str): The primary language for agent responses.
        constructor (PromptConstructor): Handles the construction of prompts for the LLM.
        llm (BaseLanguageModel): The initialized language model instance.
        tool_info (List[Dict[str, Any]]): Available tools and their configurations.
    
    Example:
        >>> agent = ChatAgent(model_name="llama", base_url="http://localhost:11434", language="english")
        >>> response = agent.respond(timestamp, sense, history, action_results, language)
        
    """
    
    def __init__(
        self, 
        model_name: str = "ollama:llama3.1:70b", 
        base_url: str = "http://localhost:11434", 
        language: str = "english"
    )->None:
        
        self.model_name = model_name
        self.base_url = base_url
        self.language = language
        
        self.constructor = PromptConstructor()
        self.tools = []
        
        # Configure model initialization parameters
        kwargs = {"temperature": 0.8}
        
        # Accommodate local model specifics
        if "ollama" in model_name:
            kwargs.update({
                "base_url" : base_url,
                "keep_alive": "1h",
                "format": "json"
            })

        try:
            self.llm = init_chat_model(model_name, **kwargs)
        except Exception as e:
            raise Exception(f"Failed to initialize model '{model_name}': {str(e)}")
            
        logger.info(f"Agent initialized with model: {self.model_name}")
    
    def set_tools(self, tools: List[Any])-> None:
        """ Set the tools for the agent """
        self.tools = tools
    
    @staticmethod
    def _format_response(message: AIMessage) -> Dict[str, Any]:
        """ Extracts thoughts, response, and tools calls from AIMessage """
        
        response_dict = {
            "analysis": "",
            "strategy": "",
            "response": "",
            "action": []
        }

        # Fallback for plain text if the model ignored tools
        if message.content and isinstance(message.content, str):
            response_dict["response"] = message.content

        # Extract tool calls
        if hasattr(message, "tool_calls"):
            for tool_call in message.tool_calls:
                name = tool_call.get("name")
                args = tool_call.get("args", {})
                
                # If it's the required RespondToUser tool, extract thoughts and speech
                if name and name.startswith("RespondToUser"):
                    response_dict["analysis"] = args.get("analysis", "")
                    response_dict["strategy"] = args.get("strategy", "")
                    response_dict["response"] = args.get("response", "")
                
                # Otherwise, it's an action tool to be executed
                else:
                    response_dict["action"].append({
                        "name": name,
                        "args": args
                    })

        return response_dict
    
    def respond(
        self,
        template: Optional[str] = None,
        timestamp: Optional[datetime] = None, 
        sense: Optional[Dict] = None, 
        history: Optional[List[Dict]] = None, 
        action_results: Optional[List[Dict]] = None, 
        language: Optional[str] = "english",
        output_format: Optional[type[BaseModel]] = None
    ) -> Dict:
        """Main response function that builds the prompt and gets a response from the language model."""
        
        timestamp = timestamp or datetime.now()
        sense = sense or {}
        history = history or []
        action_results = action_results or []
        
        prompt = self.constructor.build_prompt(
            template=template,
            timestamp=timestamp, 
            sense=sense, 
            history=history, 
            action_results=action_results
        )
        
        # We don't inject {{tools}} string into the prompt anymore,
        # but LangChain PromptTemplate requires any placeholders to be fulfilled.
        # Ensure our constructor no longer leaves an empty {{tools}} placeholder,
        # or we just format it away.
        prompt_template = PromptTemplate.from_template(prompt)
        prompt_value = prompt_template.format(tools="Available via native tool calling bindings.")
        print(f"prompt is {prompt_value}")
        
        try: 
            if output_format is not None:
                # Setup Mode: Force specific structured output
                llm_with_tools = self.llm.with_structured_output(output_format)
                response = llm_with_tools.invoke(prompt_value)
                
                # If it's a Pydantic object, return its dict. If it's already a dict, return as is.
                if isinstance(response, BaseModel):
                    return response.model_dump()
                
                return response

            else:
                # Conversation Mode: Bind arbitrary tools + RespondToUser structure
                respond_tool = RespondToUser.with_language(self.language, language)
                tools_to_bind = getattr(self, "tools", []) + [respond_tool]
                
                llm_with_tools = self.llm.bind_tools(tools_to_bind)
                ai_message = llm_with_tools.invoke(prompt_value)
                
                logger.debug(f"Raw AI Message: {ai_message}")
                return self._format_response(ai_message)
            
        except Exception as e:
            raise Exception(f"ChatAgent: Failed to get response from model: {str(e)}")
