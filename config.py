"""
Shared configuration for culture-chat.

Override any value via environment variable or by editing this file.
"""

import os

# Backend: "anthropic" or "vllm"
backend: str = os.getenv("CULTURE_CHAT_BACKEND", "anthropic")

# Model name — Claude ID for Anthropic, served model name for vLLM
model: str = os.getenv("CULTURE_CHAT_MODEL", "claude-sonnet-4-6")

# vLLM server base URL (only used when backend = "vllm")
vllm_base_url: str = os.getenv("CULTURE_CHAT_VLLM_BASE_URL", "http://localhost:8000/v1")

# Max tokens in the annotation response
max_tokens: int = int(os.getenv("CULTURE_CHAT_MAX_TOKENS", "2048"))

# Minimum number of turns a WildChat conversation must have to be annotated
min_turns: int = int(os.getenv("CULTURE_CHAT_MIN_TURNS", "6"))

# Throttling: max API requests per minute
requests_per_minute: int = 5

# Retry settings for the Anthropic API
max_retries: int = 3
retry_delay: float = 2.0  # seconds; multiplied by attempt number for rate-limit errors

# Default output directory
output_dir: str = os.getenv("CULTURE_CHAT_OUTPUT_DIR", "./output")
