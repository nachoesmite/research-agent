import sys
from pathlib import Path
from typing import Any, Dict
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from dotenv import load_dotenv
from langsmith import Client

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphs.researcher_graph import get_research_graph
from graphs.types import ResearchGraphState

def evaluate_research_agent(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Function to evaluate the research agent.
    Takes inputs from the dataset and returns outputs for evaluation.
    """
    # Create the research graph
    graph = get_research_graph()
    
    # Extract topic from inputs
    topic = inputs["topic"]
    max_analysts = inputs.get("max_analysts", 2)
    
    # Initial state
    initial_state = {
        "topic": topic,
        "max_analysts": max_analysts,
        "human_analyst_feedback": "approve"  # Auto-approve for testing
    }
    
    # Configuration
    config: Any = {"configurable": {"thread_id": f"eval-{hash(topic)}"}}
    
    try:
        # Execute the research graph
        result = graph.invoke(initial_state, config)
        
        # Return the outputs that will be evaluated
        return {
            "final_report": result.get("final_report", ""),
            "num_analysts": len(result.get("analysts", [])),
            "success": True
        }
        
    except Exception as e:
        return {
            "final_report": f"Error: {str(e)}",
            "num_analysts": 0,
            "success": False
        }
client = Client()

correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model="openai:gpt-4o",
)

def check_substantial_content(run, example):
    """Check if the report has substantial content"""
    final_report = run.outputs.get("final_report", "")
    return {
        "key": "substantial",
        "score": len(final_report) > 100
        }

def check_has_analysts(run, example):
    """Check if analysts were created"""
    return {
        "key": "has_analysts",
        "score": run.outputs.get("num_analysts", 0) > 0,
    }

def check_execution_success(run, example):
    """Check if execution was successful"""
    return {
        "key": "execution_success",
        "score": run.outputs.get("success", False)
    }

def correct(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    return outputs["output"] == reference_outputs["label"]

if __name__ == "__main__":
  print("🚀 Starting evaluation...")
  experiment_results = client.evaluate( 
      evaluate_research_agent,
      data="research-agent.evaluations",
      evaluators=[
          check_substantial_content,
          check_has_analysts,
          check_execution_success,
          correctness_evaluator #type: ignore
      ],
      experiment_prefix="research_agent_evaluation",
      max_concurrency=1,
      num_repetitions=1,
      description="Evaluating research agent performance on football topics",
      metadata={
          "version": "1.0",
          "model": "gpt-4o",
          "date": "2025-01-25"
      }
  )
