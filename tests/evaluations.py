import sys
from pathlib import Path
from typing import Any, Dict, List, Set
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
    Function to evaluate the research agent with comprehensive trajectory tracking.
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
        "human_analyst_feedback": "approve"
    }
    
    # Configuration with trajectory tracking
    config: Any = {
        "configurable": {"thread_id": f"eval-{hash(topic)}"},
        "recursion_limit": 50
    }
    
    try:
        # ✅ Capturar toda la ejecución step by step
        trajectory = []
        final_state = None
        
        # Ejecutar con stream para capturar cada step
        for step in graph.stream(initial_state, config, stream_mode="values"):
            step_info = {
                "step_number": len(trajectory) + 1,
                "state_keys": list(step.keys()),
                "has_analysts": len(step.get("analysts", [])),
                "has_sections": len(step.get("sections", [])),
                "has_final_report": bool(step.get("final_report")),
                "has_content": bool(step.get("content")),
                "has_introduction": bool(step.get("introduction")),
                "has_conclusion": bool(step.get("conclusion"))
            }
            trajectory.append(step_info)
            final_state = step
        
        # Extract nodes visited from trajectory progression (accounting for parallel execution)
        nodes_visited = []
        
        # Sequential nodes
        if any(step["has_analysts"] > 0 for step in trajectory):
            nodes_visited.append("create_analysts")
        
        if any(step["has_sections"] > 0 for step in trajectory):
            nodes_visited.append("conduct_interview")
        
        # Parallel nodes (can happen simultaneously after conduct_interview)
        if any(step["has_content"] for step in trajectory):
            nodes_visited.append("write_report")
            
        if any(step["has_introduction"] for step in trajectory):
            nodes_visited.append("write_introduction")
            
        if any(step["has_conclusion"] for step in trajectory):
            nodes_visited.append("write_conclusion")
        
        # Final sequential node
        if any(step["has_final_report"] for step in trajectory):
            nodes_visited.append("finalize_report")
        
        # Return comprehensive results
        return {
            "final_report": final_state.get("final_report", "") if final_state else "",
            "num_analysts": len(final_state.get("analysts", [])) if final_state else 0,
            "num_sections": len(final_state.get("sections", [])) if final_state else 0,
            "success": True,
            "trajectory": trajectory,
            "total_steps": len(trajectory),
            "nodes_visited": nodes_visited,
            "has_content": bool(final_state.get("content")) if final_state else False,
            "has_introduction": bool(final_state.get("introduction")) if final_state else False,
            "has_conclusion": bool(final_state.get("conclusion")) if final_state else False,
        }
        
    except Exception as e:
        return {
            "final_report": f"Error: {str(e)}",
            "num_analysts": 0,
            "num_sections": 0,
            "success": False,
            "trajectory": [],
            "total_steps": 0,
            "nodes_visited": [],
            "has_content": False,
            "has_introduction": False,
            "has_conclusion": False,
        }

client = Client()

# ✅ EVALUADORES DE TRAYECTORIA CON LÓGICA PROGRAMÁTICA

def check_follows_expected_sequence(run, example):
    """Check if nodes follow the expected graph structure (accounting for parallel execution)"""
    nodes_visited = run.outputs.get("nodes_visited", [])
    
    if not nodes_visited:
        return {"key": "follows_expected_sequence", "score": 0}
    
    # Define sequential dependencies (must be in this order)
    sequential_dependencies = [
        ("create_analysts", "conduct_interview"),  # analysts before interviews
        ("conduct_interview", "finalize_report")   # interviews before finalization
    ]
    
    # Define parallel group (can execute in any order after conduct_interview)
    parallel_group = {"write_report", "write_introduction", "write_conclusion"}
    
    score = 1.0
    
    # Check sequential dependencies
    for predecessor, successor in sequential_dependencies:
        if predecessor in nodes_visited and successor in nodes_visited:
            pred_index = nodes_visited.index(predecessor)
            succ_index = nodes_visited.index(successor)
            if pred_index >= succ_index:  # Successor should come after predecessor
                score -= 0.3
    
    # Check that parallel nodes (if present) come after conduct_interview
    if "conduct_interview" in nodes_visited:
        interview_index = nodes_visited.index("conduct_interview")
        for parallel_node in parallel_group:
            if parallel_node in nodes_visited:
                parallel_index = nodes_visited.index(parallel_node)
                if parallel_index <= interview_index:  # Should come after interview
                    score -= 0.2
    
    # Check that finalize_report (if present) comes after parallel nodes
    if "finalize_report" in nodes_visited:
        finalize_index = nodes_visited.index("finalize_report")
        for parallel_node in parallel_group:
            if parallel_node in nodes_visited:
                parallel_index = nodes_visited.index(parallel_node)
                if parallel_index >= finalize_index:  # Should come before finalize
                    score -= 0.2
    
    return {
        "key": "follows_expected_sequence",
        "score": max(0, score)  # Ensure non-negative
    }

def check_critical_nodes_coverage(run, example):
    """Check if all critical nodes were visited (accounting for parallel structure)"""
    nodes_visited = run.outputs.get("nodes_visited", [])
    
    # Define critical nodes that MUST be visited (sequential)
    essential_nodes = {"create_analysts", "conduct_interview", "finalize_report"}
    
    # Define parallel nodes (at least ONE should be visited for full workflow)
    parallel_nodes = {"write_report", "write_introduction", "write_conclusion"}
    
    # Check essential nodes coverage
    visited_essential = set(nodes_visited) & essential_nodes
    essential_score = len(visited_essential) / len(essential_nodes)
    
    # Check parallel nodes coverage (at least one)
    visited_parallel = set(nodes_visited) & parallel_nodes
    parallel_score = 1.0 if len(visited_parallel) > 0 else 0.0
    
    # Bonus for visiting more parallel nodes
    if len(visited_parallel) > 1:
        parallel_bonus = min(0.2, (len(visited_parallel) - 1) * 0.1)
        parallel_score += parallel_bonus
    
    # Combined score (essential nodes are more important)
    total_score = (essential_score * 0.7) + (parallel_score * 0.3)
    
    return {
        "key": "critical_nodes_coverage", 
        "score": min(1.0, total_score)  # Cap at 1.0
    }

def check_no_infinite_loops(run, example):
    """Check if the execution didn't get stuck in loops"""
    total_steps = run.outputs.get("total_steps", 0)
    nodes_visited = run.outputs.get("nodes_visited", [])
    
    # Simple heuristic: reasonable step count
    if total_steps == 0:
        return {"key": "no_infinite_loops", "score": 0}
    
    unique_nodes = len(set(nodes_visited)) if nodes_visited else 1
    step_to_node_ratio = total_steps / unique_nodes
    
    # Reasonable ratio threshold
    max_reasonable_ratio = 4  # Max 4 steps per unique node on average
    
    if step_to_node_ratio <= max_reasonable_ratio:
        score = 1.0
    else:
        # Penalize excessive steps
        excess_ratio = step_to_node_ratio - max_reasonable_ratio
        score = max(0, 1.0 - (excess_ratio / max_reasonable_ratio) * 0.7)
    
    return {
        "key": "no_infinite_loops",
        "score": score
    }

def check_step_efficiency(run, example):
    """Check if execution was efficient (reasonable number of steps)"""
    total_steps = run.outputs.get("total_steps", 0)
    
    # Define reasonable step ranges for research graph
    optimal_min = 4    # Minimum steps for a good execution
    optimal_max = 8    # Optimal maximum steps  
    acceptable_max = 12  # Still acceptable but not optimal
    
    if optimal_min <= total_steps <= optimal_max:
        score = 1.0
    elif total_steps < optimal_min:
        # Too few steps might mean incomplete execution
        score = total_steps / optimal_min * 0.8
    elif total_steps <= acceptable_max:
        # More steps but still acceptable
        score = 1.0 - ((total_steps - optimal_max) / (acceptable_max - optimal_max)) * 0.3
    else:
        # Too many steps - significant penalty
        score = max(0, 0.4 - ((total_steps - acceptable_max) / acceptable_max) * 0.4)
    
    return {
        "key": "step_efficiency",
        "score": score
    }

def check_complete_workflow(run, example):
    """Check if the workflow completed all phases"""
    has_content = run.outputs.get("has_content", False)
    has_introduction = run.outputs.get("has_introduction", False)
    has_conclusion = run.outputs.get("has_conclusion", False)
    final_report = run.outputs.get("final_report", "")
    
    # Check workflow completion phases
    phases_completed = 0
    total_phases = 4
    
    if run.outputs.get("num_analysts", 0) > 0:
        phases_completed += 1  # Analysts created
    
    if run.outputs.get("num_sections", 0) > 0:
        phases_completed += 1  # Interviews conducted
        
    if has_content:
        phases_completed += 1  # Report written
        
    if has_introduction and has_conclusion and len(final_report) > 100:
        phases_completed += 1  # Report finalized
    
    score = phases_completed / total_phases
    
    return {
        "key": "complete_workflow",
        "score": score
    }

def check_error_free_execution(run, example):
    """Check if execution completed without errors"""
    success = run.outputs.get("success", False)
    final_report = run.outputs.get("final_report", "")
    
    # Check if execution was successful and produced meaningful content
    has_meaningful_output = (
        len(final_report) > 50 and 
        not final_report.startswith("Error:") and
        "error" not in final_report.lower()[:100]  # No error in first 100 chars
    )
    
    score = 1 if success and has_meaningful_output else 0
    
    return {
        "key": "error_free_execution",
        "score": score
    }

def check_analyst_creation_efficiency(run, example):
    """Check if the right number of analysts were created"""
    num_analysts = run.outputs.get("num_analysts", 0)
    expected_analysts = 2  # Default expectation
    
    if num_analysts == expected_analysts:
        score = 1.0
    elif num_analysts == 0:
        score = 0.0
    elif 1 <= num_analysts <= 4:  # Reasonable range
        # Small penalty for deviation
        score = max(0.6, 1.0 - abs(num_analysts - expected_analysts) * 0.2)
    else:
        # Too many or negative analysts
        score = 0.2
    
    return {
        "key": "analyst_creation_efficiency",
        "score": score
    }

# ✅ USAR LLM SOLO PARA EVALUACIÓN DE CALIDAD DE CONTENIDO

correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model="openai:gpt-4o",
    continuous=True  # 0.0 - 1.0 score
)

def check_content_correctness(run, example):
    """Use LLM only for content quality evaluation"""
    try:
        inputs = run.inputs.get("topic", "")
        outputs = run.outputs.get("final_report", "")
        reference_outputs = ""
        
        if example and example.outputs:
            reference_outputs = example.outputs.get("expected_content", "A comprehensive research report")
        
        # Skip evaluation if no meaningful output
        if len(outputs) < 20 or outputs.startswith("Error:"):
            return {
                "key": "content_correctness",
                "score": 0
            }
        
        result = correctness_evaluator(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs
        )
        
        if isinstance(result, dict) and "score" in result:
            return {
                "key": "content_correctness",
                "score": result["score"],
                "comment": result.get("comment", "")
            }
        return {
            "key": "content_correctness",
            "score": 0
        }
        
    except Exception as e:
        print(f"Error in content evaluation: {e}")
        return {
            "key": "content_correctness",
            "score": 0
        }

def check_substantial_content(run, example):
    """Check if the report has substantial content"""
    final_report = run.outputs.get("final_report", "")
    
    # Multiple criteria for substantial content
    length_score = min(len(final_report) / 500, 1.0)  # Ideal: 500+ chars
    has_structure = final_report.count('#') >= 2  # Has sections
    has_sources = '[' in final_report and ']' in final_report  # Has citations
    
    # Combined score
    base_score = 1 if len(final_report) > 100 else 0
    if base_score:
        bonus = 0
        if has_structure:
            bonus += 0.3
        if has_sources:
            bonus += 0.2
        score = min(1.0, base_score + bonus)
    else:
        score = length_score * 0.5
    
    return {
        "key": "substantial_content",
        "score": score
    }

if __name__ == "__main__":
    print("🚀 Starting evaluation with logic-based trajectory analysis...")
    
    experiment_results = client.evaluate(  # type: ignore
        evaluate_research_agent,
        data="research-agent.evaluations",
        evaluators=[
            # ✅ Logic-based trajectory evaluators (fast & precise)
            check_follows_expected_sequence,
            check_critical_nodes_coverage,
            check_no_infinite_loops,
            check_step_efficiency,
            check_complete_workflow,
            check_error_free_execution,
            check_analyst_creation_efficiency,
            
            # ✅ LLM-based content evaluators (for quality assessment)
            check_content_correctness,
            check_substantial_content,
        ],
        experiment_prefix="research_agent_evaluation_logic_trajectory",
        max_concurrency=1,
        num_repetitions=1,
        description="Research agent evaluation with logic-based trajectory analysis and LLM content evaluation",
        metadata={
            "version": "1.0",
            "model": "gpt-4o",
            "date": "2025-01-25",
            "trajectory_evaluation": "logic_based",
            "content_evaluation": "llm_based"
        }
    )
    
    print("✅ Evaluation completed!")
    print(f"📊 Results: {experiment_results}")
