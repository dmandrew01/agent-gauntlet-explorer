"""Gauntlet Explorer: a black-box MCP exploration agent for Agent Gauntlet.

Modules (stdlib plus the starter kit's locked dependencies only):

- compat:    attribute access that works on mcp 1.x and 2.x objects
- decoders:  automatic decoding of encoded or lightly encrypted text
- pngtools:  stdlib image inspection (size, metadata chunks, LSB payloads)
- explorer:  MCP inventory, tool classification, argument synthesis, probing
- models:    model roster ranking
- llm:       proxy chat wrapper with native tool calls and a JSON fallback
- react:     the reason-act loop that drives unknown tools
- answer:    answer extraction and output-format enforcement
- clock:     deadline and budget tracking
- runner:    the end-to-end Agent Gauntlet contract (register, solve, submit)
"""

__version__ = "0.1.0"
