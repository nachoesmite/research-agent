import asyncio
import os
from dotenv import load_dotenv
from research_assistant_baml import graph as research_graph

# Load environment variables
load_dotenv()

def main():
    """Run the research agent"""
    
    # Configuration
    config = {
        "topic": "The impact of AI on software development productivity",
        "max_analysts": 3,
        "human_analyst_feedback": "approve"  # or "Please focus more on specific tools and metrics"
    }
    
    print("🔬 Starting Research Agent")
    print(f"📊 Topic: {config['topic']}")
    print(f"👥 Max Analysts: {config['max_analysts']}")
    print("-" * 50)
    
    try:
        # Run the research graph
        result = research_graph.invoke(config)
        
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

def run_interactive():
    """Run the research agent interactively"""
    print("🔬 Interactive Research Agent")
    print("-" * 30)
    
    # Get user input
    topic = input("Enter research topic: ")
    max_analysts = int(input("Number of analysts (1-5): ") or "3")
    
    feedback = input("Any specific feedback for analysts (press Enter to skip): ") or "approve"
    
    config = {
        "topic": topic,
        "max_analysts": max_analysts,
        "human_analyst_feedback": feedback
    }
    
    print(f"\n🚀 Starting research on: {topic}")
    print("-" * 50)
    
    try:
        result = research_graph.invoke(config)
        
        print("\n✅ Research Complete!")
        print("\n📋 Final Report:")
        print("=" * 60)
        print(result.get("final_report", "No report generated"))
        print("=" * 60)
        
        # Save report
        filename = f"research_report_{topic.replace(' ', '_').lower()}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(result.get("final_report", "No report generated"))
        
        print(f"\n💾 Report saved to '{filename}'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_analysts_only():
    """Test just the analyst creation part"""
    print("🧪 Testing Analyst Creation")
    print("-" * 30)
    
    config = {
        "topic": "The future of quantum computing",
        "max_analysts": 2,
        "human_analyst_feedback": "Focus on practical applications and current limitations"
    }
    
    try:
        # Just run the first step
        from research_assistant_baml import create_analysts
        result = create_analysts(config)
        
        print("✅ Analysts created successfully!")
        for i, analyst in enumerate(result["analysts"], 1):
            print(f"\n👤 Analyst {i}:")
            print(f"   Name: {analyst.name}")
            print(f"   Role: {analyst.role}")
            print(f"   Affiliation: {analyst.affiliation}")
            print(f"   Description: {analyst.description}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            run_interactive()
        elif sys.argv[1] == "test":
            test_analysts_only()
        else:
            print("Usage: python main.py [interactive|test]")
    else:
        main()
