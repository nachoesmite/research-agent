from typing import Dict, List, Any
from baml_client.types import Message as BAMLMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from baml_py import Collector
from langsmith import traceable, get_current_run_tree

def langchain_messages_to_baml(messages: List) -> List[BAMLMessage]:
    """Convert LangChain messages to BAML Message format"""
    baml_messages = []
    for msg in messages:
        # Convert content to string if it's not already
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        
        if isinstance(msg, SystemMessage):
            baml_messages.append(BAMLMessage(role="system", content=content))
        elif isinstance(msg, HumanMessage):
            baml_messages.append(BAMLMessage(role="user", content=content))
        elif isinstance(msg, AIMessage):
            name = getattr(msg, 'name', None)
            baml_messages.append(BAMLMessage(
                role="assistant", 
                content=content,
                name=name
            ))
    return baml_messages
# BAMLTracer is deprecated - use traced_client from baml_client instead