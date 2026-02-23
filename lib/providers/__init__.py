"""
lib.providers — LLM provider implementations.

Each provider module exposes a ``prompt_<provider>(...)`` function that handles
API communication, streaming, and response capture.  Currently only Claude
(Anthropic) is implemented; the package structure allows future providers
(OpenAI, Gemini, etc.) to be added as sibling modules.
"""
