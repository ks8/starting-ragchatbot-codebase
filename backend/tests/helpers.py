"""Shared mock factories for test modules."""

from unittest.mock import MagicMock


def make_text_response(text: str) -> MagicMock:
    """Create a mock Anthropic API response that returns plain text (no tool use)."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [text_block]
    return response


def make_tool_use_response(tool_name: str, tool_input: dict, tool_use_id: str = "toolu_01abc") -> MagicMock:
    """Create a mock Anthropic API response that requests tool use."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = tool_name
    tool_block.input = tool_input
    tool_block.id = tool_use_id

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_block]
    return response
