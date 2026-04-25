"""Tests for LangGraph integration."""

import pytest
from unittest.mock import Mock

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.integrations.langgraph import (
    create_nexus_node,
    create_nexus_agent,
    create_routing_node,
)


@pytest.fixture
def mock_client():
    """Create a mock Socratic Nexus client."""
    client = Mock(spec=ClaudeClient)
    client.model = "claude-3-5-sonnet-20241022"
    client.generate_response = Mock(return_value="Node response")
    return client


class TestCreateNexusNode:
    """Test create_nexus_node factory function."""

    def test_node_creation(self, mock_client):
        """Test creating a basic node."""
        node = create_nexus_node(client=mock_client, node_name="test_node")
        assert callable(node)
        assert node.__name__ == "test_node"

    def test_node_processes_state(self, mock_client):
        """Test node processes state dict correctly."""
        node = create_nexus_node(
            client=mock_client,
            node_name="analyze",
            prompt_key="content",
            output_key="analysis",
        )

        state = {"content": "Test content"}
        result = node(state)

        assert isinstance(result, dict)
        assert "analysis" in result
        assert result["analysis"] == "Node response"

    def test_node_with_system_prompt(self, mock_client):
        """Test node with system prompt prepended."""
        node = create_nexus_node(
            client=mock_client,
            node_name="summarize",
            system_prompt="You are a summarizer.",
            prompt_key="text",
            output_key="summary",
        )

        state = {"text": "Long text to summarize"}
        node(state)

        # Verify client was called with system prompt prepended
        mock_client.generate_response.assert_called_once()
        call_args = mock_client.generate_response.call_args[0][0]
        assert "You are a summarizer." in call_args
        assert "Long text to summarize" in call_args

    def test_node_with_missing_prompt(self, mock_client):
        """Test node handles missing prompt gracefully."""
        node = create_nexus_node(
            client=mock_client,
            node_name="test",
            prompt_key="missing_key",
            output_key="output",
        )

        state = {"other_key": "value"}
        result = node(state)

        # Should set output to empty string
        assert result["output"] == ""
        mock_client.generate_response.assert_not_called()

    def test_node_error_handling(self, mock_client):
        """Test node handles errors gracefully."""
        mock_client.generate_response.side_effect = Exception("API Error")

        node = create_nexus_node(
            client=mock_client,
            node_name="error_node",
            prompt_key="prompt",
            output_key="result",
        )

        state = {"prompt": "Test"}
        result = node(state)

        # Should contain error message
        assert "Error" in result["result"]

    def test_node_with_custom_keys(self, mock_client):
        """Test node with custom prompt and output keys."""
        node = create_nexus_node(
            client=mock_client,
            node_name="custom",
            prompt_key="user_input",
            output_key="assistant_response",
        )

        state = {"user_input": "Hello"}
        result = node(state)

        assert "assistant_response" in result
        assert result["assistant_response"] == "Node response"

    def test_multiple_nodes_independence(self, mock_client):
        """Test multiple nodes are independent."""
        node1 = create_nexus_node(
            client=mock_client,
            node_name="node1",
            output_key="output1",
        )
        node2 = create_nexus_node(
            client=mock_client,
            node_name="node2",
            output_key="output2",
        )

        assert node1.__name__ != node2.__name__
        assert node1 != node2


class TestCreateNexusAgent:
    """Test create_nexus_agent factory function."""

    def test_agent_creation(self, mock_client):
        """Test creating an agent."""
        agent = create_nexus_agent(
            client=mock_client,
            agent_name="test_agent",
            system_prompt="You are helpful.",
        )

        assert isinstance(agent, dict)
        assert agent["name"] == "test_agent"
        assert agent["client"] == mock_client
        assert agent["system_prompt"] == "You are helpful."
        assert callable(agent["process_fn"])

    def test_agent_process_function(self, mock_client):
        """Test agent's process function."""
        agent = create_nexus_agent(client=mock_client, system_prompt="System")
        process_fn = agent["process_fn"]

        state = {
            "messages": [
                {"role": "user", "content": "What is AI?"}
            ]
        }

        result = process_fn(state)

        assert isinstance(result, dict)
        assert "messages" in result
        # Should have added assistant message
        messages = result["messages"]
        assert len(messages) == 2
        assert messages[-1]["role"] == "assistant"

    def test_agent_with_empty_messages(self, mock_client):
        """Test agent with empty messages."""
        agent = create_nexus_agent(client=mock_client)
        process_fn = agent["process_fn"]

        state = {"messages": []}
        result = process_fn(state)

        # Should return unchanged state
        assert result == state

    def test_agent_error_handling(self, mock_client):
        """Test agent error handling."""
        mock_client.generate_response.side_effect = Exception("API Error")
        agent = create_nexus_agent(client=mock_client)
        process_fn = agent["process_fn"]

        state = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }

        result = process_fn(state)
        # Should have error message
        assert "Error" in result["messages"][-1]["content"]

    def test_agent_metadata(self, mock_client):
        """Test agent metadata."""
        agent = create_nexus_agent(
            client=mock_client,
            agent_name="claude_agent",
        )

        assert agent["name"] == "claude_agent"
        assert agent["model"] == "claude-3-5-sonnet-20241022"
        assert agent["system_prompt"] is not None


class TestCreateRoutingNode:
    """Test create_routing_node factory function."""

    def test_routing_node_creation(self, mock_client):
        """Test creating a routing node."""
        routes = {
            "summarize": "Summarize the text",
            "analyze": "Analyze the text",
            "generate": "Generate new content",
        }
        node = create_routing_node(
            client=mock_client,
            node_name="router",
            routes=routes,
        )

        assert callable(node)
        assert node.__name__ == "router"

    def test_routing_node_decision(self, mock_client):
        """Test routing node makes routing decisions."""
        routes = {
            "short": "Short response",
            "long": "Long response",
        }
        mock_client.generate_response.return_value = "short"

        node = create_routing_node(
            client=mock_client,
            node_name="router",
            prompt_key="input",
            output_key="next_node",
            routes=routes,
        )

        state = {"input": "What should I do?"}
        result = node(state)

        assert "next_node" in result
        assert result["next_node"] == "short"

    def test_routing_node_without_routes(self, mock_client):
        """Test routing node without predefined routes."""
        node = create_routing_node(
            client=mock_client,
            node_name="router",
            prompt_key="query",
            output_key="action",
        )

        state = {"query": "Process this"}
        result = node(state)

        assert "action" in result
        # Should contain the LLM response
        assert result["action"] == "Node response"

    def test_routing_node_error_handling(self, mock_client):
        """Test routing node error handling."""
        mock_client.generate_response.side_effect = Exception("Error")

        node = create_routing_node(
            client=mock_client,
            node_name="router",
            output_key="route",
        )

        state = {"content": "test"}
        result = node(state)

        assert result["route"] == "error"

    def test_routing_with_strips_whitespace(self, mock_client):
        """Test routing node strips whitespace."""
        mock_client.generate_response.return_value = "  summarize  \n"

        node = create_routing_node(
            client=mock_client,
            node_name="router",
            output_key="route",
        )

        state = {"input": "test"}
        result = node(state)

        assert result["route"] == "summarize"


class TestNodeIntegration:
    """Integration tests for nodes working together."""

    def test_sequential_node_execution(self, mock_client):
        """Test sequential execution of multiple nodes."""
        mock_client.generate_response.side_effect = [
            "Analysis result",
            "Summary result",
            "Final result",
        ]

        analyze_node = create_nexus_node(
            client=mock_client,
            node_name="analyze",
            prompt_key="content",
            output_key="analysis",
        )
        summarize_node = create_nexus_node(
            client=mock_client,
            node_name="summarize",
            prompt_key="analysis",
            output_key="summary",
        )

        # Execute nodes sequentially
        state = {"content": "Test content"}
        state = analyze_node(state)
        assert state["analysis"] == "Analysis result"

        state = summarize_node(state)
        assert state["summary"] == "Summary result"

    def test_node_state_preservation(self, mock_client):
        """Test that nodes preserve existing state."""
        node = create_nexus_node(
            client=mock_client,
            node_name="processor",
            prompt_key="input",
            output_key="output",
        )

        state = {
            "input": "test",
            "metadata": {"key": "value"},
            "other_field": 42,
        }

        result = node(state)

        # All original fields should be preserved
        assert result["metadata"] == {"key": "value"}
        assert result["other_field"] == 42
        assert result["input"] == "test"
        assert "output" in result

    def test_agent_conversation_flow(self, mock_client):
        """Test multi-turn agent conversation."""
        agent = create_nexus_agent(client=mock_client)
        process_fn = agent["process_fn"]

        state = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }

        # First turn
        state = process_fn(state)
        assert len(state["messages"]) == 2

        # Add another user message
        state["messages"].append(
            {"role": "user", "content": "Follow up"}
        )

        # Second turn
        state = process_fn(state)
        assert len(state["messages"]) == 4
        assert state["messages"][-1]["role"] == "assistant"
