"""Tests for data models."""

from socratic_nexus.models import (
    TokenUsage,
    ChatResponse,
    ProjectContext,
    ConflictInfo,
)


def test_token_usage_creation():
    """Test TokenUsage model creation."""
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        provider="claude",
        model="claude-3-sonnet",
        cost_usd=0.01,
        latency_ms=500.0,
    )
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150
    assert usage.provider == "claude"
    assert usage.model == "claude-3-sonnet"
    assert usage.cost_usd == 0.01
    assert usage.latency_ms == 500.0


def test_token_usage_defaults():
    """Test TokenUsage defaults."""
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        provider="openai",
        model="gpt-4",
    )
    assert usage.cost_usd == 0.0
    assert usage.latency_ms == 0.0


def test_chat_response_creation():
    """Test ChatResponse model creation."""
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        provider="claude",
        model="claude-3-sonnet",
    )
    response = ChatResponse(
        content="Hello, world!",
        usage=usage,
        model="claude-3-sonnet",
        provider="claude",
        stop_reason="end_turn",
        metadata={"custom": "data"},
    )
    assert response.content == "Hello, world!"
    assert response.usage == usage
    assert response.model == "claude-3-sonnet"
    assert response.provider == "claude"
    assert response.stop_reason == "end_turn"
    assert response.metadata == {"custom": "data"}


def test_chat_response_defaults():
    """Test ChatResponse defaults."""
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        provider="claude",
        model="claude-3-sonnet",
    )
    response = ChatResponse(
        content="Hello",
        usage=usage,
        model="claude-3-sonnet",
        provider="claude",
    )
    assert response.stop_reason is None
    assert response.metadata == {}


def test_project_context_creation():
    """Test ProjectContext model creation."""
    context = ProjectContext(
        project_name="Test Project",
        description="A test project",
        files={"main.py": "print('hello')"},
        metadata={"version": "1.0"},
        goals=["Goal 1", "Goal 2"],
        phase="planning",
        tech_stack=["Python", "FastAPI"],
    )
    assert context.project_name == "Test Project"
    assert context.description == "A test project"
    assert context.files == {"main.py": "print('hello')"}
    assert context.metadata == {"version": "1.0"}
    assert context.goals == ["Goal 1", "Goal 2"]
    assert context.phase == "planning"
    assert context.tech_stack == ["Python", "FastAPI"]


def test_project_context_defaults():
    """Test ProjectContext defaults."""
    context = ProjectContext(project_name="Test")
    assert context.project_name == "Test"
    assert context.description == ""
    assert context.files == {}
    assert context.metadata == {}
    assert context.goals is None
    assert context.phase is None
    assert context.tech_stack is None


def test_conflict_info_creation():
    """Test ConflictInfo model creation."""
    conflict = ConflictInfo(
        description="Database schema conflict",
        file_path="schema.sql",
        line_number=42,
        resolution_options=["Option A", "Option B"],
        metadata={"severity": "high"},
        conflict_type="schema_change",
        old_value="VARCHAR(100)",
        new_value="TEXT",
        old_author="alice",
        new_author="bob",
        severity="high",
    )
    assert conflict.description == "Database schema conflict"
    assert conflict.file_path == "schema.sql"
    assert conflict.line_number == 42
    assert conflict.resolution_options == ["Option A", "Option B"]
    assert conflict.conflict_type == "schema_change"
    assert conflict.old_value == "VARCHAR(100)"
    assert conflict.new_value == "TEXT"
    assert conflict.old_author == "alice"
    assert conflict.new_author == "bob"
    assert conflict.severity == "high"


def test_conflict_info_defaults():
    """Test ConflictInfo defaults."""
    conflict = ConflictInfo(description="Test conflict")
    assert conflict.description == "Test conflict"
    assert conflict.file_path == ""
    assert conflict.line_number == 0
    assert conflict.resolution_options == []
    assert conflict.metadata == {}
    assert conflict.conflict_type is None
    assert conflict.old_value is None
    assert conflict.new_value is None
