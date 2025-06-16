from abc import ABC, abstractmethod
from typing import List, Iterable
from llm_adapter import LLMChunk # need to define this?



class BaseAdapter(ABC):
    """Common interface for chat completions with tool support."""
    
    @abstractmethod
    def generate(self, messages: List[dict]) -> Iterable[LLMChunk]:
        """Yield chunks with .role, .text, .function_call optional."""
