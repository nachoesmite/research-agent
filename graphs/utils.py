from typing import Dict, List
from baml_client.types import Message as BAMLMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from baml_client import b
from baml_py import Collector
from langsmith import traceable, get_current_run_tree
from langsmith import traceable, get_current_run_tree

from typing import Any

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
class BAMLTracer:
    def __init__(self):
        self.client = b
    
    def call_baml_function(self, function_name: str, **kwargs) -> Any:
        """Llamada pura a BAML con collector para extraer datos"""
        
        # 1. Llamar a BAML con collector
        collector = Collector(name=f"{function_name.lower()}-collector")
        kwargs["baml_options"] = {"collector": collector}
        
        baml_function = getattr(self.client, function_name)
        result = baml_function(**kwargs)
        
        # 2. Extraer input y output del collector
        llm_input_messages = None
        llm_output_messages = None
        usage_metadata = {}
        print(collector.last)
        if collector.last and collector.last.calls and len(collector.last.calls) > 0:
            http_request = collector.last.calls[0].http_request
            http_response = collector.last.calls[0].http_response

            if hasattr(http_request, 'body') and hasattr(http_request.body, 'json'):
                llm_input_messages = http_request.body.json()
            if hasattr(http_response, 'body') and hasattr(http_response.body, 'json'):
                llm_output_messages = http_response.body.json()
       # 3. Llamar a función traceada con raw input/output
        self._trace_llm_call(
            function_name=function_name,
            raw_input=llm_input_messages or [],
            raw_output=llm_output_messages or [],
            usage=usage_metadata
        )
        
        return result
    
    @traceable(
        run_type="llm", 
        metadata={"ls_provider": "baml", "ls_model_name": "gpt-4o"}
    )
    def _trace_llm_call(self, function_name: str, raw_input: List, raw_output: Any, usage: Dict):
        """Función traceada que recibe solo el raw input y output del LLM"""
        
        # El decorador @traceable captura automáticamente estos inputs/outputs
        # raw_input = mensajes reales enviados al LLM
        # raw_output = respuesta real del LLM
        
        run = get_current_run_tree()
        if run:
            # Setear usage metadata usando el método oficial
            if usage:
                run.set(usage_metadata={
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                })
            
            # Metadata adicional
            run.extra = {"baml_function": function_name}
            run.tags = [f"baml:{function_name}", "llm:gpt-4o"]
        
        return raw_output