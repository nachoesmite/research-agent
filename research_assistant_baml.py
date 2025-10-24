import operator
from pydantic import BaseModel, Field
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string

from langgraph.constants import Send
from langgraph.graph import END, MessagesState, START, StateGraph

# Import BAML client
from baml_client import b
from baml_client.types import Message as BAMLMessage, Analyst, Perspectives, SearchQuery

### Schema - Using BAML types

class GenerateAnalystsState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions

class InterviewState(MessagesState):
    max_num_turns: int # Number turns of conversation
    context: Annotated[list, operator.add] # Source docs
    analyst: Analyst # Analyst asking questions
    interview: str # Interview transcript
    sections: list # Final key we duplicate in outer state for Send() API

class ResearchGraphState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions
    sections: Annotated[list, operator.add] # Send() API key
    introduction: str # Introduction for the final report
    content: str # Content for the final report
    conclusion: str # Conclusion for the final report
    final_report: str # Final report

### Helper functions

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

def get_analyst_persona(analyst: Analyst) -> str:
    """Get analyst persona string"""
    return f"Name: {analyst.name}\nRole: {analyst.role}\nAffiliation: {analyst.affiliation}\nDescription: {analyst.description}\n"

### Nodes and edges

def create_analysts(state: GenerateAnalystsState):
    """Create analysts using BAML"""
    
    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')
    
    # Use BAML function to generate analysts
    perspectives = b.CreateAnalysts(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts
    )
    
    # Write the list of analysts to state
    return {"analysts": perspectives.analysts}

def human_feedback(state: GenerateAnalystsState):
    """No-op node that should be interrupted on"""
    pass

def generate_question(state: InterviewState):
    """Node to generate a question using BAML"""

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    
    # Convert messages to BAML format
    baml_messages = langchain_messages_to_baml(messages)
    
    # Generate question using BAML
    analyst_persona = get_analyst_persona(analyst)
    question_content = b.GenerateQuestion(
        analyst_persona=analyst_persona,
        messages=baml_messages
    )
    
    # Create AI message
    question = AIMessage(content=question_content)
    
    # Write messages to state
    return {"messages": [question]}

def search_web(state: InterviewState):
    """Retrieve docs from web search"""

    # Search
    tavily_search = TavilySearchResults(max_results=3)

    # Convert messages to BAML format
    baml_messages = langchain_messages_to_baml(state['messages'])
    
    # Generate search query using BAML
    search_query_result = b.GenerateSearchQuery(messages=baml_messages)
    
    # Search
    search_docs = tavily_search.invoke(search_query_result.search_query)

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

def search_wikipedia(state: InterviewState):
    """Retrieve docs from wikipedia"""

    # Convert messages to BAML format
    baml_messages = langchain_messages_to_baml(state['messages'])
    
    # Generate search query using BAML
    search_query_result = b.GenerateSearchQuery(messages=baml_messages)
    
    # Search
    search_docs = WikipediaLoader(
        query=search_query_result.search_query, 
        load_max_docs=2
    ).load()

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

def generate_answer(state: InterviewState):
    """Node to answer a question using BAML"""

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    # Generate answer using BAML
    analyst_persona = get_analyst_persona(analyst)
    context_str = "\n\n".join(context)
    
    answer_content = b.GenerateAnswer(
        analyst_persona=analyst_persona,
        context=context_str
    )
    
    # Create AI message
    answer = AIMessage(content=answer_content)
    answer.name = "expert"
    
    # Append it to state
    return {"messages": [answer]}

def save_interview(state: InterviewState):
    """Save interviews"""

    # Get messages
    messages = state["messages"]
    
    # Convert interview to a string
    interview = get_buffer_string(messages)
    
    # Save to interviews key
    return {"interview": interview}

def route_messages(state: InterviewState, name: str = "expert"):
    """Route between question and answer"""
    
    # Get messages
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns', 2)

    # Check the number of expert answers 
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # This router is run after each question - answer pair 
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]
    
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"

def write_section(state: InterviewState):
    """Node to write a section using BAML"""

    # Get state
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]
   
    # Write section using BAML
    context_str = "\n\n".join(context)
    section_result = b.WriteSection(
        analyst_description=analyst.description,
        context=context_str
    )
    
    # Append it to state
    return {"sections": [section_result.content]}

# Add nodes and edges 
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# Flow
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages, ['ask_question', 'save_interview'])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

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
    report_content = b.WriteReport(
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
    intro_content = b.WriteIntroduction(
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
    conclusion_content = b.WriteConclusion(
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

# Add nodes and edges 
builder = StateGraph(ResearchGraphState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_builder.compile())
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
graph = builder.compile(interrupt_before=['human_feedback'])