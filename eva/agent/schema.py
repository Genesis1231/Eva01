from config import validate_language, logger
from pydantic import BaseModel, Field, create_model
from typing import Any, List, Dict, Type
from functools import lru_cache

class RespondToUser(BaseModel):
    """How I think and speak. I always use this to express myself."""
    feeling: str = Field(description="What I feel right now.", max_length=20)
    inner_monologue: str = Field(description="What I notice and think about this, in my own words.")
    response: str = Field(description="What I say out loud.")

class SetupDesireOutput(BaseModel):
    """Output format for the name retrieval"""
    
    analysis: str = Field(description="My reflection and analysis")
    strategy: str = Field(description="My response strategy")
    response: str = Field(description="My verbal response")
    desire: str = Field(description="The most important thing the user desire in life within two words.")
    confidence: float = Field(description="My confidence level in the retrieved desire, from 0 to 1")