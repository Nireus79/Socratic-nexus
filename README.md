# Socratic Nexus (socrates-nexus)

Universal LLM abstraction layer providing unified access to multiple AI providers

## Overview

socrates-nexus is a specialized component of the Socratic ecosystem that provides comprehensive functionality for universal llm abstraction layer providing unified access to multiple ai providers. It offers a production-ready, well-tested solution for integrating into larger Socratic systems.

## Key Features

- Multi-provider support
- Unified client interface
- Async/sync API support
- Model standardization

## Installation

### Prerequisites
- Python 3.8+
- pip or poetry package manager

### Via pip

```bash
pip install socrates-nexus
```

### Via poetry

```bash
poetry add socrates-nexus
```

### From source

```bash
git clone https://github.com/socratic-ai/socrates-nexus.git
cd socrates-nexus
pip install -e .
```

## Quick Start

### Basic Usage

```python
from socrates_nexus import Client

# Initialize the component
client = Client()

# Perform basic operation
result = client.process()
print(result)
```

### Configuration

```python
from socrates_nexus import Client

# Configure with custom settings
client = Client(
    api_key="your-api-key",
    model="claude-3-5-haiku",
    timeout=30
)

# Use the client
response = client.query("What is the essence of learning?")
```

## Core Components

- **Client**: Main interface for interacting
- **Providers**: Provider-specific implementations
- **Models**: Model definitions and configurations
- **Async Client**: Asynchronous version
- **Exceptions**: Custom error handling

## Architecture Overview

socrates-nexus follows a modular, layered architecture:

- **Client Layer**: Public API for external consumers
- **Processing Layer**: Core business logic and algorithms
- **Integration Layer**: Connections to external services and APIs
- **Data Layer**: State management and data persistence

## Integration with Socratic Ecosystem

socrates-nexus integrates seamlessly with:

- **socrates-nexus**: Unified LLM provider interface
- **socratic-rag**: Knowledge retrieval and augmentation
- **socratic-core**: Common utilities and base classes
- **Other ecosystem components**: As needed for specific use cases

## Dependencies

```
- anthropic
- openai
- google-generativeai
- langchain
```

## Performance Considerations

- **Caching**: Intelligent caching for frequently accessed data
- **Async Support**: Full async/await support for non-blocking I/O
- **Scalability**: Designed for high-concurrency scenarios
- **Resource Efficiency**: Optimized memory and CPU usage

## Error Handling

The library provides comprehensive error handling:

```python
from socrates_nexus import Client, Error

try:
    client = Client()
    result = client.process()
except Error as e:
    print(f"Error: {e}")
```

## Testing

Run the test suite:

```bash
pytest tests/
```

With coverage reporting:

```bash
pytest --cov=socrates_nexus tests/
```

## API Reference

For complete API documentation, see [ARCHITECTURE.md](ARCHITECTURE.md)

## Contributing

Contributions are welcome! See CONTRIBUTING.md for guidelines.

## Support & Resources

- GitHub Issues: https://github.com/socratic-ai/socrates-nexus/issues
- Documentation: https://socratic-docs.readthedocs.io/socrates-nexus
- Discussions: https://github.com/socratic-ai/socrates-nexus/discussions

## License

MIT License - See LICENSE file for details

---

Part of the Socratic Ecosystem
