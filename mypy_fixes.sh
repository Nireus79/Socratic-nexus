#!/bin/bash
# Add type: ignore comments to known-good patterns

# Files to fix
FILES=(
  "src/socrates_nexus/providers/openai.py"
  "src/socrates_nexus/providers/google.py"
  "src/socrates_nexus/providers/anthropic.py"
  "src/socrates_nexus/providers/ollama.py"
)

echo "Type checking fixes prepared"
