"""
Example 15: LangGraph Integration

Demonstrates how to use Socratic Nexus clients as nodes in LangGraph
workflows for building stateful multi-step AI applications.

Installation:
    pip install socratic-nexus langgraph
"""

import os
from socratic_nexus.clients import ClaudeClient
from socratic_nexus.integrations.langgraph import (
    create_nexus_node,
    create_nexus_agent,
    create_routing_node,
)

print("=" * 60)
print("LANGGRAPH INTEGRATION - NODE FACTORY EXAMPLE")
print("=" * 60)

# Create a Socratic Nexus client
client = ClaudeClient(api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-..."))

print("\n1. Creating a Nexus node:")
print("-" * 40)

# Create a node that analyzes text
analysis_node = create_nexus_node(
    client=client,
    node_name="analyze_text",
    prompt_key="content",
    output_key="analysis",
    system_prompt="You are a text analysis expert. Analyze the provided text.",
)

print(f"Created node: {analysis_node.__name__}")
print(f"Node docstring: {analysis_node.__doc__}\n")

# Example state
state = {
    "content": "Machine learning is a subset of artificial intelligence.",
    "analysis": "",
}

print("2. Running the node:")
print("-" * 40)
result_state = analysis_node(state)
print(f"Analysis result: {result_state['analysis'][:100]}...\n")

# Example with routing node
print("3. Creating a routing node:")
print("-" * 40)

router = create_routing_node(
    client=client,
    node_name="intent_router",
    prompt_key="input",
    output_key="next_node",
    routes={
        "summarize": "Summarize the text",
        "analyze": "Analyze the text for insights",
        "generate": "Generate new content based on the text",
    },
)

print(f"Created routing node: {router.__name__}\n")

# Example with agent
print("4. Creating a Nexus agent:")
print("-" * 40)

agent_config = create_nexus_agent(
    client=client,
    agent_name="chat_agent",
    system_prompt="You are a helpful AI assistant.",
)

print(f"Agent name: {agent_config['name']}")
print(f"Agent model: {agent_config['model']}")
print(f"Agent has process function: {callable(agent_config['process_fn'])}\n")

# Example LangGraph workflow (conceptual)
print("=" * 60)
print("LANGGRAPH WORKFLOW STRUCTURE")
print("=" * 60)

print("""
With LangGraph, you can build workflows like:

from langgraph.graph import StateGraph

class WorkflowState(TypedDict):
    input: str
    analysis: str
    summary: str
    output: str

graph = StateGraph(WorkflowState)

# Add nodes
graph.add_node("analyze", analysis_node)
graph.add_node("route", router)

# Add edges
graph.add_edge("start", "analyze")
graph.add_edge("analyze", "route")

# Conditional routing
graph.add_conditional_edges(
    "route",
    lambda state: state["next_node"]
)

# Compile and run
workflow = graph.compile()
result = workflow.invoke({"input": "..."})
""")

print("\nKey features:")
print("- create_nexus_node: Create nodes for LangGraph workflows")
print("- create_nexus_agent: Create agents with state management")
print("- create_routing_node: Create dynamic routing decisions")
print("- State dict compatibility for easy integration")

print("\nFor more information:")
print("- LangGraph docs: https://langchain-ai.github.io/langgraph")
print("- Socratic Nexus: https://github.com/Nireus79/Socratic-nexus")
