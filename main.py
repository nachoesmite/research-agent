import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from graphs.researcher_graph import get_research_graph

# Load environment variables
load_dotenv()

def main():
    """Run the research agent"""
    
    # Configuration
    config = {
        "topic": "The impact of AI on software development productivity",
        "max_analysts": 2,
    }
    
    print("🔬 Starting Research Agent with BAML")
    print(f"📊 Topic: {config['topic']}")
    print(f"👥 Max Analysts: {config['max_analysts']}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    try:
        # Get the compiled research graph
        research_graph = get_research_graph()
        
        # Run the research graph with thread support for interrupts
        from langgraph.graph.state import CompiledStateGraph
        
        thread_id = "research_session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # First invoke - will pause at human_feedback if needed
        result = research_graph.invoke(
            config, 
            {"configurable": {"thread_id": thread_id}}
        )

        current_state = research_graph.get_state({"configurable": {"thread_id": thread_id}})
        
        print(f"🔍 Current state:")
        print(f"   Next node: {current_state.next}")
        print(f"   Values keys: {list(current_state.values.keys()) if current_state.values else 'None'}")
        print(f"   Metadata: {current_state.metadata}")        
        # Check if we're interrupted (result will be None)
        if current_state.next and 'human_feedback' in current_state.next:
            print("\n⏸️  Paused for human feedback...")
            
            # Get current state to show analysts
            state = research_graph.get_state({"configurable": {"thread_id": thread_id}})
            if state.values and "analysts" in state.values:
                analysts = state.values["analysts"]
                print("Created analysts:")
                for i, analyst in enumerate(analysts, 1):
                    print(f"\n👤 Analyst {i}: {analyst.name}")
                    print(f"   Role: {analyst.role}")
                    print(f"   Focus: {analyst.description[:100]}...")
            
            feedback = input("\nProvide feedback (or 'approve' to continue): ").strip()
            if not feedback:
                feedback = "approve"
            
            print(f"📝 Feedback: {feedback}")
            
            # Update state with feedback and resume
            research_graph.update_state(
                {"configurable": {"thread_id": thread_id}}, 
                {"human_analyst_feedback": feedback}
            )
            result = research_graph.invoke(
                None, 
                {"configurable": {"thread_id": thread_id}}
            )
        
        print("\n✅ Research Complete!")
        print("\n📋 Final Report:")
        print("=" * 60)
        print(result.get("final_report", "No report generated"))
        print("=" * 60)
        
        # Save report to file
        with open("research_report.md", "w", encoding="utf-8") as f:
            f.write(result.get("final_report", "No report generated"))
        
        print("\n💾 Report saved to 'research_report.md'")
        
    except Exception as e:
        print(f"❌ Error running research agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
