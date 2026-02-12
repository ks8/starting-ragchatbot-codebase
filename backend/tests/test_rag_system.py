"""Tests for RAGSystem.query() end-to-end orchestration."""

import pytest
from unittest.mock import MagicMock, patch, call
from tests.helpers import make_text_response, make_tool_use_response


def _make_rag_system():
    """Create a RAGSystem with all external dependencies mocked."""
    with patch('rag_system.VectorStore') as MockVS, \
         patch('rag_system.AIGenerator') as MockAI, \
         patch('rag_system.DocumentProcessor') as MockDP:

        from config import Config
        config = Config()
        config.ANTHROPIC_API_KEY = "test-key"

        from rag_system import RAGSystem
        rag = RAGSystem(config)

    return rag


class TestRAGSystemQuery:

    def test_query_wraps_user_input(self):
        """ai_generator.generate_response receives query wrapped with prefix."""
        rag = _make_rag_system()
        rag.ai_generator.generate_response.return_value = "Some answer"

        rag.query("What is MCP?")

        call_kwargs = rag.ai_generator.generate_response.call_args.kwargs
        assert call_kwargs["query"] == "Answer this question about course materials: What is MCP?"

    def test_query_passes_tools_and_manager(self):
        """generate_response is called with tools list and tool_manager."""
        rag = _make_rag_system()
        rag.ai_generator.generate_response.return_value = "Answer"

        rag.query("test query")

        call_kwargs = rag.ai_generator.generate_response.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] is rag.tool_manager

    def test_query_returns_response_and_sources(self):
        """Returns a (response_text, sources_list) tuple."""
        rag = _make_rag_system()
        rag.ai_generator.generate_response.return_value = "The answer"

        result = rag.query("test")

        assert isinstance(result, tuple)
        assert len(result) == 2
        response, sources = result
        assert response == "The answer"
        assert isinstance(sources, list)

    def test_query_resets_sources_after_retrieval(self):
        """After query() returns, tool_manager sources are cleared."""
        rag = _make_rag_system()
        rag.ai_generator.generate_response.return_value = "Answer"

        # Manually populate sources on the search tool to simulate a search
        for tool in rag.tool_manager.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = ["Source 1", "Source 2"]

        _, sources = rag.query("test")

        # Sources should have been retrieved
        assert sources == ["Source 1", "Source 2"]
        # And then reset
        assert rag.tool_manager.get_last_sources() == []

    def test_query_with_session_gets_history(self):
        """When session_id provided, conversation history is fetched and passed."""
        rag = _make_rag_system()
        rag.ai_generator.generate_response.return_value = "Answer"

        # Create a session and add some history
        session_id = rag.session_manager.create_session()
        rag.session_manager.add_exchange(session_id, "prev question", "prev answer")

        rag.query("new question", session_id=session_id)

        call_kwargs = rag.ai_generator.generate_response.call_args.kwargs
        assert call_kwargs["conversation_history"] is not None
        assert "prev question" in call_kwargs["conversation_history"]

    def test_query_without_session_no_history(self):
        """When session_id is None, conversation_history is None."""
        rag = _make_rag_system()
        rag.ai_generator.generate_response.return_value = "Answer"

        rag.query("test", session_id=None)

        call_kwargs = rag.ai_generator.generate_response.call_args.kwargs
        assert call_kwargs["conversation_history"] is None

    def test_query_exception_propagates(self):
        """Exception from generate_response bubbles up (becomes 500 in app.py)."""
        rag = _make_rag_system()
        rag.ai_generator.generate_response.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            rag.query("test")

    def test_query_updates_session_history(self):
        """After query, session_manager has the new exchange."""
        rag = _make_rag_system()
        rag.ai_generator.generate_response.return_value = "The response"

        session_id = rag.session_manager.create_session()
        rag.query("my question", session_id=session_id)

        history = rag.session_manager.get_conversation_history(session_id)
        assert "my question" in history
        assert "The response" in history
