"""
Shared configuration for culture-chat.

Override any value via environment variable or by editing this file.
"""

import os

# Anthropic model used for annotation
model: str = os.getenv("CULTURE_CHAT_MODEL", "claude-opus-4-6")

# Max tokens in the annotation response
max_tokens: int = int(os.getenv("CULTURE_CHAT_MAX_TOKENS", "2048"))

# Minimum number of turns a WildChat conversation must have to be annotated
min_turns: int = int(os.getenv("CULTURE_CHAT_MIN_TURNS", "6"))

# Retry settings for the Anthropic API
max_retries: int = 3
retry_delay: float = 2.0  # seconds; multiplied by attempt number for rate-limit errors

# Default output directory
output_dir: str = os.getenv("CULTURE_CHAT_OUTPUT_DIR", "./output")
