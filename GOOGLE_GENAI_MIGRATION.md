# Migration to google-genai Package

## Overview
Migrated from the deprecated `google.generativeai` package to the new `google-genai` package to address the deprecation warning and ensure compatibility with future Google AI releases.

## What Changed

### Package Migration
- **Old:** `google-generativeai>=0.3.0` (deprecated, no longer maintained)
- **New:** `google-genai>=1.0.0` (official, actively maintained)

### API Changes

#### 1. Client Initialization
**Before:**
```python
import google.generativeai as genai
genai.configure(api_key="your-api-key")
model = genai.GenerativeModel("gemini-pro")
response = model.generate_content("prompt")
```

**After:**
```python
import google.genai as genai
client = genai.Client(api_key="your-api-key")
response = client.models.generate_content(model="gemini-2.0-flash", contents="prompt")
```

#### 2. Content Generation
- **Old:** `model.generate_content(prompt)` - returns response with `.text` attribute
- **New:** `client.models.generate_content(model=name, contents=prompt)` - same return format

#### 3. Async Operations
- **Old:** Used `asyncio.to_thread()` for async (no native async support)
- **New:** Native async with `client.aio.models.generate_content()`

### Model Updates
- Updated default model from `gemini-pro` to `gemini-2.0-flash` (more capable, better performance)

## Files Modified

### Source Code
- `src/socratic_nexus/clients/google_client.py`
  - Changed import from `google.generativeai` to `google.genai`
  - Updated `__init__()` to use `genai.Client()` directly
  - Changed all API calls to use `client.models.generate_content()`
  - Updated async calls to use `client.aio.models.generate_content()`
  - Removed `genai.configure()` calls

### Configuration
- `pyproject.toml`
  - Updated dependency from `google-generativeai>=0.3.0` to `google-genai>=1.0.0`
  - Version bumped to 0.4.0

### Tests
- `tests/test_other_clients_execution.py`
  - Updated mocks from `genai.GenerativeModel` to `genai.Client`
  - Changed mock method from `client.generate_content()` to `client.models.generate_content()`

## Benefits

1. **Future-Proof:** Using actively maintained, official package
2. **No Deprecation Warnings:** Eliminates warning about deprecated google.generativeai
3. **Better Async:** Native async support instead of thread workarounds
4. **Improved Models:** Default to gemini-2.0-flash which is more capable
5. **Cleaner API:** More explicit API with separate model specification

## Installation

For users upgrading socratic-nexus:
```bash
pip install socratic-nexus>=0.4.0
```

The new google-genai package will be installed automatically.

## Compatibility

- **Breaking Change:** Yes - requires google-genai instead of google-generativeai
- **Python Support:** 3.9+
- **Backward Compatibility:** No - clients using old google.generativeai API will need to update

## Testing

All tests updated and passing. Coverage maintained at expected levels.

## Version History

- **0.3.7:** Last version using google.generativeai
- **0.4.0:** First version using google-genai (current)

