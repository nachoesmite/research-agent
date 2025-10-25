from baml_client.types import Analyst
from graphs.types import GenerateAnalystsState, InterviewState
from graphs.traced_client import traced_client
from graphs.utils import langchain_messages_to_baml
from langchain_core.messages import AIMessage, get_buffer_string
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

def get_analyst_persona(analyst: Analyst) -> str:
    """Get analyst persona string"""
    return f"Name: {analyst.name}\nRole: {analyst.role}\nAffiliation: {analyst.affiliation}\nDescription: {analyst.description}\n"

### Nodes and edges

def create_analysts(state: GenerateAnalystsState):
    """Create analysts using BAML"""
    
    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')
    
    # Use BAML function to generate analysts with automatic tracing
    perspectives = traced_client.CreateAnalysts(
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
    question_content = traced_client.GenerateQuestion(
        analyst_persona=analyst_persona,
        messages=baml_messages
    )
    
    # Create AI message
    question = AIMessage(content=question_content)
    
    # Write messages to state
    return {"messages": [question]}

def search_web(state: InterviewState):
    """Retrieve docs from web search"""

    tavily_search = TavilySearchResults(max_results=3)

    # Convert messages to BAML format
    baml_messages = langchain_messages_to_baml(state['messages'])
    
    # Generate search query using BAML
    search_query_result = traced_client.GenerateSearchQuery(messages=baml_messages)
    
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
    search_query_result = traced_client.GenerateSearchQuery(messages=baml_messages)
    
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
    baml_messages = langchain_messages_to_baml(state['messages'])
    context = state["context"]

    # Generate answer using BAML
    analyst_persona = get_analyst_persona(analyst)
    context_str = "\n\n".join(context)
    
    answer_content = traced_client.GenerateAnswer(
        analyst_persona=analyst_persona,
        context=context_str
        messages=baml_messages
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
    section_result = traced_client.WriteSection(
        analyst_description=analyst.description,
        context=context_str
    )
    
    # Append it to state
    return {"sections": [section_result.content]}

def get_interview_graph() -> CompiledStateGraph:
    

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

  return interview_builder.compile()