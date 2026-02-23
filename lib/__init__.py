"""
lib — modular package for the ai-code tool.

Each sub-module isolates a single concern (config loading, file discovery,
PDF extraction, Claude API interaction, etc.).  The entry point ``ai-code.py``
imports from these modules and orchestrates the workflow.

No module in this package executes logic on import — all initialisation flows
through explicit function calls from ``main()``.
"""
