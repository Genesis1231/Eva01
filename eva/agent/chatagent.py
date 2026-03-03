"""
ChatAgent: A chat agent that manages interactions with the user.

Handles initialization, configuration, and inference with different language models (LLMs).
Manages prompt construction, response formatting, and tool integration.

"""
    
from config import logger
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain.chat_models import init_chat_model

from .schema import RespondToUser
from .constructor import PromptConstructor

class ChatAgent:
    """
    ChatAgent class:
    
    Attributes:
        model_selection (str): The identifier for the selected language model (e.g., "LLAMA", "GPT").
        base_url (str): The base URL for API connections, defaults to localhost for local models.
        language (str): The primary language for agent responses.
        constructor (PromptConstructor): Handles the construction of prompts for the LLM.
        llm (BaseLanguageModel): The initialized language model instance.
        tool_info (List[Dict[str, Any]]): Available tools and their configurations.
    
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
    
    def _build_prompt_value(
        self,
        template: Optional[str],
        timestamp: datetime,
        sense: Dict,
        history: List[Dict],
        action_results: List[Dict],
    ) -> str:
        """Build and format the prompt string."""
        prompt = self.constructor.build_prompt(
            template=template,
            timestamp=timestamp,
            sense=sense,
            history=history,
            action_results=action_results,
        )
        prompt_template = PromptTemplate.from_template(prompt)
        return prompt_template.format(tools="Available via native tool calling bindings.")

    async def arespond(
        self,
        template: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        sense: Optional[Dict] = None,
        history: Optional[List[Dict]] = None,
        action_results: Optional[List[Dict]] = None,
        language: Optional[str] = "english",
        output_format: Optional[type[BaseModel]] = None,
    ) -> Dict:
        """Async response — same as respond() but non-blocking. Use inside async loops."""
        timestamp = timestamp or datetime.now()
        sense = sense or {}
        history = history or []
        action_results = action_results or []

        prompt_value = self._build_prompt_value(template, timestamp, sense, history, action_results)

        try:
            if output_format is not None:
                llm_with_tools = self.llm.with_structured_output(output_format)
                response = await llm_with_tools.ainvoke(prompt_value)
                return response.model_dump() if isinstance(response, BaseModel) else response

            respond_tool = RespondToUser.with_language(self.language, language)
            llm_with_tools = self.llm.bind_tools(getattr(self, "tools", []) + [respond_tool])
            ai_message = await llm_with_tools.ainvoke(prompt_value)
            logger.debug(f"Raw AI Message: {ai_message}")
            return self._format_response(ai_message)

        except Exception as e:
            raise Exception(f"ChatAgent: Failed to get response from model: {str(e)}")
