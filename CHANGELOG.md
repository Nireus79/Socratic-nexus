# Changelog

All notable changes to Socrates Nexus will be documented in this file.

## [0.3.5] - 2025-04-25

### ✨ Production Ready Release

#### Code Quality & Testing
- **Test Coverage**: Achieved 70.94% coverage (target exceeded)
- **New Test Suites**: Added 80+ tests across 4 new test files
  - `test_all_llm_clients.py` - Comprehensive client tests
  - `test_clients_error_handling.py` - Error scenario coverage
  - `test_clients_methods.py` - Method implementation tests
  - `test_coverage_gaps.py` - Coverage-focused tests
- **Code Quality**: Fixed all ruff linting violations
- **Formatting**: Applied Black formatting across all files
- **Type Safety**: Fixed mypy errors and improved type hints

#### Improvements
- **Type Hints**: Better use of `Optional[]` for clearer contracts
- **Documentation**: Improved README and architecture documentation
- **Bug Fixes**: Fixed lowercase `any` type hints (should be `Any`)
- **Compatibility**: Enhanced support for orchestrator-based deployments

#### Cleanup
- **Repository**: Removed 15+ obsolete documentation and temporary files
- **Build Artifacts**: Cleaned up test coverage files
- **Duplicates**: Removed duplicate documentation files

#### PyPI Preparation
- **Metadata**: Enhanced pyproject.toml with complete classifiers
- **Documentation**: Professional README with comprehensive examples
- **Architecture**: Detailed ARCHITECTURE.md with design patterns
- **Security**: Added SECURITY.md guidelines

### 🔧 Technical Details

#### Claude Client Enhancements
- 9 specialized generation methods for domain-specific tasks
- Full async/sync support with consistent interface
- Accurate token cost tracking
- Intelligent response caching

#### All Clients Maintain
- Unified interface across all providers
- Complete async/sync variants
- Token usage tracking and cost calculation
- Comprehensive error handling
- Graceful authentication handling

---

## [0.3.0] - 2025-03-10

### 🎉 Major Features (Phase 2 Complete)

#### Vision Models Support
- **Multimodal Content**: Added support for image analysis across all providers
- **ImageContent Model**: New dataclass for handling image data (URLs, file paths, base64)
- **Provider Support**: Vision models for Anthropic Claude 3.5 Sonnet, OpenAI GPT-4 Vision, and Google Gemini
- **Image Utilities**: New `socrates_nexus.utils.images` module
- **API**: Simple - pass `images` parameter to `chat()` method

#### Function Calling / Tool Use
- **Tool Definitions**: New `Tool`, `Function`, `ToolCall`, `FunctionCall` models
- **Provider Support**: Anthropic Claude, OpenAI GPT-4, and Google Gemini
- **Tool Choice Modes**: Support for `auto`, `none`, and specific tool selection
- **Schema Support**: Full JSON Schema parameter definitions

### 🧪 Testing & Quality (Phase 2 Test Coverage)

- **Total Tests**: 381 new tests added (216 unit tests + 165 integration/edge case tests)
- **Coverage**: Increased from 35% to **76% coverage**
- **Provider Tests**: BaseProvider (40), Anthropic (23), OpenAI (14), Google (11), Ollama (14)
- **Feature Tests**: Streaming (46), Client Integration (31), Edge Cases (37), Vision & Tools (28)
- **CI/CD**: 100% workflows passing with coverage threshold enforcement

### 🔧 CI/CD Improvements

- **Smart Test Matrix**: Fast PR testing, comprehensive main testing
- **Quality Checks**: Separate workflows for lint, type-check, security scanning
- **Coverage Enforcement**: Automatic failure if coverage drops below 70%
- **Dependabot**: Automated dependency updates

### 📚 Documentation

- **Vision Models Guide** (`docs/vision.md`) - Complete guide with examples and troubleshooting
- **Function Calling Guide** (`docs/function-calling.md`) - Complete guide with best practices
- **README.md** - Updated with Vision and Function Calling sections
- **Coverage Badge**: Updated to 76%
- **Support Email**: Hermes_creative@proton.me

### 📦 Examples

- **`examples/12_vision_models.py`** - Vision example with URLs, local files, multiple images, provider comparison
- **`examples/13_function_calling.py`** - Function calling example with tool definitions, multiple tools, provider comparison

### 🔄 Changes to Existing APIs

- **ChatResponse**: Added `tool_calls` field and `content_blocks` for structured content
- **Message Handling**: Support for both simple strings and structured `MessageContent` objects
- **Backward Compatible**: All existing code continues to work unchanged

### 📋 Notes

- All Phase 1 functionality remains unchanged
- New vision and function calling features are opt-in
- Future: Streaming with tool calls, Tool registry, Extended model support

---

## [0.2.0] - 2025-03-05

### ✅ Phase 1: Foundation Complete

- Universal LLM client supporting Claude, GPT-4, Gemini, Llama
- Streaming support for all providers
- Automatic retry logic with exponential backoff
- Token tracking and cost calculation
- Response caching with TTL
- Multi-provider fallback patterns

---

## [0.1.0] - 2025-02-28

### Initial Release
- Basic LLM client functionality
- Foundation for Phase 1 development

---

## Support

For issues, questions, or feedback:
- **GitHub Issues**: [Report bugs](https://github.com/Nireus79/socrates-nexus/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/Nireus79/socrates-nexus/discussions)
- **Email**: Hermes_creative@proton.me
