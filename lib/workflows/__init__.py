"""
lib.workflows — high-level workflow orchestrators.

Each workflow module exposes a ``run_*_workflow(cfg, args)`` function that
composes tools from ``lib.tools`` into a complete end-to-end flow.  The
entry point ``ai-code.py`` delegates to these workflows based on CLI flags.

Workflows:
    workflow_ai          — standard single-shot prompt execution (-ai)
    workflow_gen_source   — source list generation (-gen-source)
    workflow_ai_steps     — automated multi-step pipeline (-ai-steps)
"""
