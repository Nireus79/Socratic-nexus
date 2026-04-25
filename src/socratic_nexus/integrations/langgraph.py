"""
LangGraph integration for Socratic Nexus clients.

Provides utilities to use Socratic Nexus clients as nodes in LangGraph workflows.

Example:
    from socratic_nexus.clients import ClaudeClient
    from socratic_nexus.integrations.langgraph import create_nexus_node
    from langgraph.graph import StateGraph

    client = ClaudeClient(api_key="your-api-key")

    # Create a node that uses the Socratic Nexus client
    analysis_node = create_nexus_node(
        client=client,
        node_name="analyze",
        prompt_key="content",
        output_key="analysis"
    )

    # Use in a LangGraph workflow
    graph = StateGraph(MyState)
    graph.add_node("analyze", analysis_node)
    graph.add_edge("start", "analyze")
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, TypeVar

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.clients.google_client import GoogleClient
from socratic_nexus.clients.openai_client import OpenAIClient
from socratic_nexus.clients.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Type variable for state objects
StateType = TypeVar("StateType")


def create_nexus_node(
    client: ClaudeClient | OpenAIClient | GoogleClient | OllamaClient,
    node_name: str = "nexus_node",
    prompt_key: str = "input",
    output_key: str = "output",
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Callable[[StateType], StateType]:
    """
    Create a LangGraph node that uses a Socratic Nexus client.

    The returned node function takes the current state, extracts a prompt
    from it, passes it to the Socratic Nexus client, and updates the state
    with the result.

    Args:
        client: The Socratic Nexus client to use
        node_name: Name of the node (for logging)
        prompt_key: Key in state dict to use as the prompt
        output_key: Key in state dict to store the output
        system_prompt: Optional system prompt to prepend
        temperature: Temperature for generation
        max_tokens: Maximum tokens in response

    Returns:
        A callable node function compatible with LangGraph
    """

    def node_fn(state: StateType) -> StateType:
        """
        Execute the node with the given state.

        Args:
            state: The current state (typically a dict)

        Returns:
            Updated state with the node's output
        """
        try:
            # Extract prompt from state
            if isinstance(state, dict):
                prompt = state.get(prompt_key, "")
            else:
                prompt = getattr(state, prompt_key, "")

            if not prompt:
                logger.warning(f"Node {node_name}: No prompt found in state[{prompt_key}]")
                if isinstance(state, dict):
                    state = dict(state)
                else:
                    state = vars(state)
                state[output_key] = ""
                return state

            # Prepare full prompt with system message if provided
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            logger.debug(f"Node {node_name}: Processing prompt ({len(prompt)} chars)")

            # Call the client
            response = client.generate_response(full_prompt)

            # Extract text from response
            if isinstance(response, str):
                output = response
            elif hasattr(response, "content"):
                output = response.content
            else:
                output = str(response)

            # Update state with output
            if isinstance(state, dict):
                state = dict(state)
            else:
                state = dict(vars(state))

            state[output_key] = output
            logger.debug(f"Node {node_name}: Generated output ({len(output)} chars)")

            return state

        except Exception as e:
            logger.error(f"Error in node {node_name}: {e}")
            if isinstance(state, dict):
                state = dict(state)
            else:
                state = dict(vars(state))
            state[output_key] = f"Error: {str(e)}"
            return state

    # Store metadata on the function for debugging
    node_fn.__name__ = node_name  # type: ignore[attr-defined]
    node_fn.__doc__ = f"LangGraph node using Socratic Nexus {type(client).__name__}"  # type: ignore[attr-defined]

    return node_fn


def create_nexus_agent(
    client: ClaudeClient | OpenAIClient | GoogleClient | OllamaClient,
    agent_name: str = "nexus_agent",
    system_prompt: str = "You are a helpful assistant.",
    tools: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Create a LangGraph agent using a Socratic Nexus client.

    Returns a dictionary with agent configuration that can be used with
    LangGraph's agent framework.

    Args:
        client: The Socratic Nexus client to use
        agent_name: Name of the agent
        system_prompt: System prompt for the agent
        tools: Optional list of tools (not yet implemented)

    Returns:
        Dictionary with agent configuration
    """

    def process_message(state: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message through the agent."""
        messages = state.get("messages", [])

        if not messages:
            return state

        # Get the latest message
        latest_message = messages[-1]
        if isinstance(latest_message, dict):
            prompt = latest_message.get("content", "")
        else:
            prompt = getattr(latest_message, "content", "")

        # Generate response
        try:
            response = client.generate_response(f"{system_prompt}\n\nUser: {prompt}")

            if isinstance(response, str):
                output = response
            elif hasattr(response, "content"):
                output = response.content
            else:
                output = str(response)

            # Add response to messages
            new_messages = list(messages)
            new_messages.append({"role": "assistant", "content": output})

            state = dict(state)
            state["messages"] = new_messages

            return state

        except Exception as e:
            logger.error(f"Error in agent {agent_name}: {e}")
            new_messages = list(messages)
            new_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
            state = dict(state)
            state["messages"] = new_messages
            return state

    return {
        "name": agent_name,
        "client": client,
        "system_prompt": system_prompt,
        "process_fn": process_message,
        "model": getattr(client, "model", "unknown"),
    }


def create_routing_node(
    client: ClaudeClient,
    node_name: str = "router",
    prompt_key: str = "content",
    output_key: str = "next_node",
    routes: Optional[Dict[str, str]] = None,
) -> Callable:
    """
    Create a routing node that uses Claude to determine the next step.

    This node can be used to dynamically route to different branches
    in a LangGraph workflow based on the input.

    Args:
        client: Claude client to use for routing decisions
        node_name: Name of the node
        prompt_key: Key in state to use as input
        output_key: Key in state to store the routing decision
        routes: Optional dict of {route_name: description}

    Returns:
        A callable node function
    """

    def router_fn(state: Any) -> Any:
        """Route to the next node based on LLM decision."""
        try:
            # Extract content from state
            if isinstance(state, dict):
                content = state.get(prompt_key, "")
            else:
                content = getattr(state, prompt_key, "")

            # Create routing prompt
            if routes:
                routes_str = "\n".join([f"- {name}: {desc}" for name, desc in routes.items()])
                routing_prompt = (
                    f"Given the following input, determine which of these routes to take:\n"
                    f"{routes_str}\n\n"
                    f"Input: {content}\n\n"
                    f"Respond with only the route name."
                )
            else:
                routing_prompt = (
                    f"Analyze this input and determine the next appropriate step: {content}"
                )

            # Get routing decision
            response = client.generate_response(routing_prompt)

            if isinstance(response, str):
                next_node = response.strip()
            elif hasattr(response, "content"):
                next_node = response.content.strip()
            else:
                next_node = str(response).strip()

            # Update state
            if isinstance(state, dict):
                state = dict(state)
            else:
                state = dict(vars(state))

            state[output_key] = next_node

            return state

        except Exception as e:
            logger.error(f"Error in routing node {node_name}: {e}")
            if isinstance(state, dict):
                state = dict(state)
            else:
                state = dict(vars(state))
            state[output_key] = "error"
            return state

    router_fn.__name__ = node_name  # type: ignore[attr-defined]
    return router_fn


__all__ = ["create_nexus_node", "create_nexus_agent", "create_routing_node"]
