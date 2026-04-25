"""
Integrations for Socratic Nexus with popular frameworks.

Supported frameworks:
- LangChain: Use Socratic Nexus clients as LLM providers in LangChain chains
- LangGraph: Use Socratic Nexus clients as nodes in LangGraph workflows
- Openclaw: Use Socratic Nexus clients as Openclaw skills
"""

from __future__ import annotations

__all__ = []

# LangChain integration
try:
    from .langchain import SocratesNexusLLM  # noqa: F401

    __all__.append("SocratesNexusLLM")
except ImportError:
    pass

# LangGraph integration
try:
    from .langgraph import (  # noqa: F401
        create_nexus_node,
        create_nexus_agent,
    )

    __all__.extend(["create_nexus_node", "create_nexus_agent"])
except ImportError:
    pass

# Openclaw integration
try:
    from .openclaw import NexusLLMSkill  # noqa: F401

    __all__.append("NexusLLMSkill")
except ImportError:
    pass
