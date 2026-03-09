"""
Example: Using Socrates Nexus with LangChain

Demonstrates how to use Socrates Nexus as a LangChain-compatible LLM provider.

Installation:
    pip install socrates-nexus[langchain]
"""

from socrates_nexus.integrations.langchain import SocratesNexusLLM


def example_basic_chain():
    """Basic LangChain chain with Socrates Nexus."""
    print("=== Basic LangChain Chain ===\n")

    try:
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
    except ImportError:
        print("LangChain not installed. Install with: pip install langchain")
        return

    llm = SocratesNexusLLM(provider="anthropic", model="claude-opus")
    prompt = PromptTemplate(template="Explain {topic} in one sentence", input_variables=["topic"])
    chain = LLMChain(llm=llm, prompt=prompt)

    result = chain.run(topic="machine learning")
    print(f"Topic: machine learning")
    print(f"Response: {result}\n")


def example_multi_provider_chains():
    """Compare responses from multiple providers using same chain."""
    print("=== Multi-Provider Comparison ===\n")

    try:
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
    except ImportError:
        print("LangChain not installed. Install with: pip install langchain")
        return

    prompt = PromptTemplate(
        template="What is {topic}? Answer in 2 sentences.", input_variables=["topic"]
    )

    providers = [
        ("anthropic", "claude-opus"),
        ("openai", "gpt-4"),
        ("google", "gemini-pro"),
    ]

    topic = "artificial intelligence"

    for provider, model in providers:
        print(f"{provider.upper()} ({model}):")
        try:
            llm = SocratesNexusLLM(provider=provider, model=model)
            chain = LLMChain(llm=llm, prompt=prompt)
            result = chain.run(topic=topic)
            print(f"  {result[:120]}...\n")
        except Exception as e:
            print(f"  [Error] {str(e)[:100]}\n")


def example_qa_chain():
    """Question-answering chain with multiple providers."""
    print("=== Q&A Chain ===\n")

    try:
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
    except ImportError:
        print("LangChain not installed. Install with: pip install langchain")
        return

    qa_template = """Answer the following question based on your knowledge:

Question: {question}

Answer:"""

    llm = SocratesNexusLLM(provider="anthropic", model="claude-opus")
    prompt = PromptTemplate(template=qa_template, input_variables=["question"])
    chain = LLMChain(llm=llm, prompt=prompt)

    questions = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "How many planets are in our solar system?",
    ]

    for q in questions:
        result = chain.run(question=q)
        print(f"Q: {q}")
        print(f"A: {result.strip()}\n")


def example_provider_fallback():
    """Fallback chain that tries multiple providers."""
    print("=== Provider Fallback Pattern ===\n")

    try:
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
    except ImportError:
        print("LangChain not installed. Install with: pip install langchain")
        return

    providers = [
        ("anthropic", "claude-opus"),
        ("openai", "gpt-4"),
        ("google", "gemini-pro"),
    ]

    prompt = PromptTemplate(
        template="Write a creative name for a {product} company", input_variables=["product"]
    )

    product = "coffee shop"

    for provider, model in providers:
        print(f"Trying {provider} ({model})...")
        try:
            llm = SocratesNexusLLM(provider=provider, model=model)
            chain = LLMChain(llm=llm, prompt=prompt)
            result = chain.run(product=product)
            print(f"Success! Generated name: {result.strip()}\n")
            break
        except Exception as e:
            print(f"Failed: {str(e)[:80]}")
            print(f"Trying next provider...\n")


def example_chain_with_config():
    """Chain with custom configuration."""
    print("=== Chain with Custom Config ===\n")

    try:
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
    except ImportError:
        print("LangChain not installed. Install with: pip install langchain")
        return

    llm = SocratesNexusLLM(
        provider="anthropic",
        model="claude-opus",
        temperature=0.9,  # More creative
        max_tokens=500,
        retry_attempts=3,
        cache_responses=True,
    )

    prompt = PromptTemplate(
        template="Tell me a creative story about {character}", input_variables=["character"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    result = chain.run(character="a robot learning to paint")
    print(f"Story:\n{result}\n")


if __name__ == "__main__":
    print("Socrates Nexus - LangChain Integration Examples\n")
    print("=" * 50 + "\n")

    example_basic_chain()
    example_multi_provider_chains()
    example_qa_chain()
    example_provider_fallback()
    example_chain_with_config()

    print("=" * 50)
    print("Examples complete!")
