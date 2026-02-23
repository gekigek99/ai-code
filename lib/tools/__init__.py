"""
lib.tools — reusable workflow tool functions.

Each tool module exposes a single primary function that performs one
well-defined operation (source generation, prompt execution, expansion,
step decomposition, or user confirmation).  Tools accept a Config object
and return structured result dicts with at minimum a ``status`` key.

Tools are stateless — no module-level globals.  They may call the LLM,
read/write files, or interact with the user, but each such side effect is
explicitly documented in the function's docstring.
"""
