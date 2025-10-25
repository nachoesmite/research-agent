from graphs.types import ResearchGraphState
from graphs.interview_graph import create_analysts, human_feedback, get_interview_graph
from graphs.traced_client import traced_client
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

def initiate_all_interviews(state: ResearchGraphState):
    """Conditional edge to initiate all interviews via Send() API or return to create_analysts"""    

    # Check if human feedback
    human_analyst_feedback = state.get('human_analyst_feedback', 'approve')
    if human_analyst_feedback.lower() != 'approve':
        # Return to create_analysts
        return "create_analysts"

    # Otherwise kick off interviews in parallel via Send() API
    else:
        topic = state["topic"]
        return [Send("conduct_interview", {
            "analyst": analyst,
            "messages": [HumanMessage(
                content=f"So you said you were writing an article on {topic}?"
            )]
        }) for analyst in state["analysts"]]

def write_report(state: ResearchGraphState):
    """Node to write the final report body using BAML"""

    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Generate report using BAML
    report_content = traced_client.WriteReport(
        topic=topic,
        sections=formatted_str_sections
    )
    
    return {"content": report_content}

def write_introduction(state: ResearchGraphState):
    """Node to write the introduction using BAML"""

    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Generate introduction using BAML
    intro_content = traced_client.WriteIntroduction(
        topic=topic,
        sections=formatted_str_sections
    )
    
    return {"introduction": intro_content}

def write_conclusion(state: ResearchGraphState):
    """Node to write the conclusion using BAML"""

    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Generate conclusion using BAML
    conclusion_content = traced_client.WriteConclusion(
        topic=topic,
        sections=formatted_str_sections
    )
    
    return {"conclusion": conclusion_content}

def finalize_report(state: ResearchGraphState):
    """The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion"""

    # Save full final report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}

def get_research_graph() -> CompiledStateGraph:
    builder = StateGraph(ResearchGraphState)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("conduct_interview", get_interview_graph())
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)

    # Logic
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")
    builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
    builder.add_edge("finalize_report", END)

    # Compile
    memory = MemorySaver()
    # interrupt_before=['human_feedback'], 
    return builder.compile(checkpointer=memory)