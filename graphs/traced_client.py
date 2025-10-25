# Traced BAML Client - A wrapper that adds tracing to BAML functions
from typing import Any, Dict, List, Optional
from baml_py import Collector
from langsmith import traceable, get_current_run_tree
from baml_client.sync_client import BamlSyncClient
from baml_client import b

class TracedBamlClient:
    """
    BAML client wrapper with automatic LangSmith tracing.
    
    This client automatically wraps ALL methods from the underlying BamlSyncClient
    with tracing, so it adapts automatically when new BAML functions are added.
    """
    
    def __init__(self, client: Optional[BamlSyncClient] = None):
        self.client = client or b
    
    def __getattr__(self, name: str):
        """
        Automatically wrap any BAML function call with tracing.
        This means new BAML functions work immediately without code changes.
        """
        # Check if the method exists on the underlying client
        if hasattr(self.client, name):
            original_method = getattr(self.client, name)
            
            # If it's a callable method, wrap it with tracing
            if callable(original_method):
                def traced_wrapper(*args, **kwargs):
                    return self.llm_call(name, *args, **kwargs)
                return traced_wrapper
            else:
                # If it's a property, return it directly
                return original_method
        
        # If method doesn't exist, raise AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def llm_call(self, function_name: str, *args, **kwargs) -> Any:
        """Internal method that handles tracing for all BAML calls"""
        collector = Collector(name=f"{function_name.lower()}-collector")
        kwargs["baml_options"] = {"collector": collector}
        
        baml_function = getattr(self.client, function_name)
        result = baml_function(*args, **kwargs)
        
        llm_input_messages = None
        llm_output_messages = None
        
        if collector.last and collector.last.calls and len(collector.last.calls) > 0:
            http_request = collector.last.calls[0].http_request
            http_response = collector.last.calls[0].http_response

            if http_request and hasattr(http_request, 'body') and http_request.body and hasattr(http_request.body, 'json'):
                llm_input_messages = http_request.body.json()
                print(llm_input_messages)
            if http_response and hasattr(http_response, 'body') and http_response.body and hasattr(http_response.body, 'json'):
                llm_output_messages = http_response.body.json()
                print(llm_output_messages)
                
        self._trace_llm_call(
            function_name=function_name,
            raw_input=llm_input_messages or [],
            raw_output=llm_output_messages or [],
        )
        
        return result
    
    @traceable(
        run_type="llm", 
        metadata={"ls_provider": "baml", "ls_model_name": "gpt-4o"}
    )
    def _trace_llm_call(self, function_name: str, raw_input: List, raw_output: Any):
        """Función traceada que recibe solo el raw input y output del LLM"""
        run = get_current_run_tree()
        if run:
            # Setear usage metadata usando el método oficial
            # if usage:
            #     run.set(usage_metadata={
            #         "input_tokens": usage.get("prompt_tokens", 0),
            #         "output_tokens": usage.get("completion_tokens", 0),
            #         "total_tokens": usage.get("total_tokens", 0)
            #     })
            
            # Metadata adicional
            run.extra = {"baml_function": function_name}
            run.tags = [f"baml:{function_name}", "llm:gpt-4o"]
            run.name = f"BAML {function_name}"
        return raw_output

# Create a global traced client instance
traced_client = TracedBamlClient()