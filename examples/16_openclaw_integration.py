"""
Example 16: Openclaw Integration

Demonstrates how to use Socratic Nexus clients as Openclaw skills
for building composable AI applications.

Installation:
    pip install socratic-nexus
"""

import os
from socratic_nexus.clients import ClaudeClient
from socratic_nexus.integrations.openclaw import (
    NexusLLMSkill,
    NexusAnalysisSkill,
    NexusCodeGenSkill,
    NexusDocumentationSkill,
)

print("=" * 60)
print("OPENCLAW INTEGRATION - SKILL FACTORY EXAMPLE")
print("=" * 60)

# Create a Socratic Nexus client
client = ClaudeClient(api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-..."))

print("\n1. Creating a basic NexusLLMSkill:")
print("-" * 40)

# Create a basic LLM skill
llm_skill = NexusLLMSkill(
    client=client,
    name="text_generator",
    description="Generate text using Claude",
)

print(f"Skill: {llm_skill}")
print(f"Model: {llm_skill.model}")
print(f"Name: {llm_skill.name}\n")

# Query the skill
print("2. Querying the skill:")
print("-" * 40)

response = llm_skill.query("What is the capital of France?")
print(f"Response: {response[:100]}...\n")

# Get skill info
print("3. Getting skill information:")
print("-" * 40)

info = llm_skill.get_info()
print(f"Skill info:")
for key, value in info.items():
    print(f"  {key}: {value}\n")

# Specialized skills
print("4. Using specialized skills:")
print("-" * 40)

# Analysis skill
analysis_skill = NexusAnalysisSkill(client=client)
print(f"Created analysis skill: {analysis_skill.name}")
analysis_result = analysis_skill.analyze("Artificial intelligence is transforming industries.")
print(f"Analysis: {analysis_result['analysis'][:100]}...\n")

# Code generation skill
codegen_skill = NexusCodeGenSkill(client=client)
print(f"Created code generation skill: {codegen_skill.name}")
code_result = codegen_skill.generate_code("Function to calculate factorial", language="python")
print(f"Generated code:\n{code_result['code'][:150]}...\n")

# Documentation skill
docs_skill = NexusDocumentationSkill(client=client)
print(f"Created documentation skill: {docs_skill.name}")
docs_result = docs_skill.generate_documentation("REST API design", doc_type="guide")
print(f"Documentation:\n{docs_result['documentation'][:100]}...\n")

# Process interface (for structured workflows)
print("5. Using the process interface:")
print("-" * 40)

input_data = {
    "prompt": "What is machine learning?",
    "context": "You are an expert educator",
    "instructions": "Explain in simple terms for beginners",
}

process_result = llm_skill.process(input_data)
print(f"Process result:")
print(f"  Success: {process_result['success']}")
print(f"  Output: {process_result['output'][:100]}...")
print(f"  Metadata: {process_result['metadata']}\n")

print("=" * 60)
print("OPENCLAW INTEGRATION COMPLETE")
print("=" * 60)

print("\nKey features:")
print("- NexusLLMSkill: Basic LLM skill with query interface")
print("- NexusAnalysisSkill: Specialized for text analysis")
print("- NexusCodeGenSkill: Specialized for code generation")
print("- NexusDocumentationSkill: Specialized for documentation")
print("- Process interface for structured workflows")
print("- Metadata and info retrieval for skill registration")

print("\nSkill capabilities:")
print("- Text generation and analysis")
print("- Code generation and review")
print("- Documentation generation")
print("- Question answering")
print("- Custom system prompts")

print("\nFor more information:")
print("- Openclaw docs: https://github.com/openclaw/openclaw")
print("- Socratic Nexus: https://github.com/Nireus79/Socratic-nexus")
