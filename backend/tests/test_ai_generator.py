"""Tests for AIGenerator tool-calling behavior."""

import pytest
from unittest.mock import MagicMock, patch
from ai_generator import AIGenerator
from tests.helpers import make_text_response, make_tool_use_response


def _make_generator():
    """Create an AIGenerator with a mocked Anthropic client."""
    with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        generator = AIGenerator(api_key="test-key", model="test-model")
    # generator.client is now mock_client
    return generator, mock_client


class TestGenerateResponse:

    def test_direct_text_response(self):
        """When API returns stop_reason='end_turn', returns text directly with no tool execution."""
        generator, mock_client = _make_generator()
        mock_client.messages.create.return_value = make_text_response("Hello world")

        result = generator.generate_response(query="Hi")

        assert result == "Hello world"
        mock_client.messages.create.assert_called_once()

    def test_tools_included_in_api_call(self):
        """When tools provided, api_params includes 'tools' and tool_choice='auto'."""
        generator, mock_client = _make_generator()
        mock_client.messages.create.return_value = make_text_response("Answer")

        tools = [
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {},
            }
        ]
        generator.generate_response(query="test", tools=tools)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    def test_no_tools_excludes_tool_params(self):
        """When no tools provided, api_params does not include 'tools' or 'tool_choice'."""
        generator, mock_client = _make_generator()
        mock_client.messages.create.return_value = make_text_response("Answer")

        generator.generate_response(query="test")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs

    def test_tool_use_triggers_execution(self):
        """When API returns stop_reason='tool_use', tool_manager.execute_tool is called."""
        generator, mock_client = _make_generator()
        mock_client.messages.create.side_effect = [
            make_tool_use_response("search_course_content", {"query": "MCP"}),
            make_text_response("MCP is a protocol."),
        ]

        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "search results here"

        result = generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        mock_tm.execute_tool.assert_called_once_with(
            "search_course_content", query="MCP"
        )
        assert result == "MCP is a protocol."

    def test_tool_result_sent_back_to_api(self):
        """Second API call includes tool_result messages with correct tool_use_id."""
        generator, mock_client = _make_generator()
        mock_client.messages.create.side_effect = [
            make_tool_use_response(
                "search_course_content", {"query": "MCP"}, tool_use_id="toolu_123"
            ),
            make_text_response("Final answer"),
        ]

        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "tool output"

        generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        # Second call should have tool results in messages
        second_call = mock_client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]

        # Last message should be the tool results (role=user with tool_result content)
        tool_result_msg = messages[-1]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "toolu_123"
        assert tool_result_msg["content"][0]["content"] == "tool output"

    def test_follow_up_call_includes_tools(self):
        """Follow-up API call after first tool execution includes 'tools' (not final round)."""
        generator, mock_client = _make_generator()
        mock_client.messages.create.side_effect = [
            make_tool_use_response("search_course_content", {"query": "MCP"}),
            make_text_response("Final answer"),
        ]

        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "results"

        generator.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        assert "tools" in second_call_kwargs

    def test_conversation_history_in_system(self):
        """When conversation_history provided, system prompt includes it."""
        generator, mock_client = _make_generator()
        mock_client.messages.create.return_value = make_text_response("Answer")

        history = "User: Hi\nAssistant: Hello"
        generator.generate_response(query="test", conversation_history=history)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "Previous conversation:" in call_kwargs["system"]
        assert history in call_kwargs["system"]

    def test_no_history_system_prompt_only(self):
        """When no conversation_history, system is just SYSTEM_PROMPT."""
        generator, mock_client = _make_generator()
        mock_client.messages.create.return_value = make_text_response("Answer")

        generator.generate_response(query="test")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == AIGenerator.SYSTEM_PROMPT

    def test_no_tool_manager_with_tool_use_response(self):
        """When stop_reason='tool_use' but tool_manager=None, falls through to content[0].text.
        This is a potential bug — ToolUseBlock has no .text attribute."""
        generator, mock_client = _make_generator()
        tool_response = make_tool_use_response(
            "search_course_content", {"query": "MCP"}
        )
        mock_client.messages.create.return_value = tool_response

        # This should attempt response.content[0].text on a tool_use block.
        # MagicMock won't raise (it auto-creates attributes), but real SDK would fail.
        result = generator.generate_response(query="test", tool_manager=None)

        # With MagicMock, .text is auto-created — verify the code path was taken
        # (only 1 API call, no tool execution)
        mock_client.messages.create.assert_called_once()

    def test_api_exception_propagates(self):
        """Exception from messages.create propagates unhandled."""
        generator, mock_client = _make_generator()
        mock_client.messages.create.side_effect = Exception("API connection failed")

        with pytest.raises(Exception, match="API connection failed"):
            generator.generate_response(query="test")

    def test_second_api_call_failure(self):
        """Exception from the second messages.create (after tool results) propagates."""
        generator, mock_client = _make_generator()
        mock_client.messages.create.side_effect = [
            make_tool_use_response("search_course_content", {"query": "MCP"}),
            Exception("Rate limited"),
        ]

        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "results"

        with pytest.raises(Exception, match="Rate limited"):
            generator.generate_response(
                query="test",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tm,
            )

    def test_two_sequential_tool_calls(self):
        """Happy path: outline → search → text. Both tools executed, 3 API calls."""
        generator, mock_client = _make_generator()
        mock_client.messages.create.side_effect = [
            make_tool_use_response(
                "get_course_outline", {"course_name": "AI"}, tool_use_id="toolu_1"
            ),
            make_tool_use_response(
                "search_course_content",
                {"query": "transformers"},
                tool_use_id="toolu_2",
            ),
            make_text_response("Transformers are covered in lesson 3."),
        ]

        mock_tm = MagicMock()
        mock_tm.execute_tool.side_effect = ["outline data", "search results"]

        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]
        result = generator.generate_response(
            query="test", tools=tools, tool_manager=mock_tm
        )

        assert result == "Transformers are covered in lesson 3."
        assert mock_tm.execute_tool.call_count == 2
        assert mock_client.messages.create.call_count == 3

    def test_final_round_excludes_tools(self):
        """Third API call (final round) does NOT contain 'tools'; second call DOES."""
        generator, mock_client = _make_generator()
        mock_client.messages.create.side_effect = [
            make_tool_use_response(
                "get_course_outline", {"course_name": "AI"}, tool_use_id="toolu_1"
            ),
            make_tool_use_response(
                "search_course_content",
                {"query": "transformers"},
                tool_use_id="toolu_2",
            ),
            make_text_response("Final answer"),
        ]

        mock_tm = MagicMock()
        mock_tm.execute_tool.side_effect = ["outline data", "search results"]

        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]
        generator.generate_response(query="test", tools=tools, tool_manager=mock_tm)

        second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        assert "tools" in second_call_kwargs
        assert second_call_kwargs["tool_choice"] == {"type": "auto"}

        third_call_kwargs = mock_client.messages.create.call_args_list[2].kwargs
        assert "tools" not in third_call_kwargs
        assert "tool_choice" not in third_call_kwargs

    def test_early_termination_no_second_tool_call(self):
        """Claude returns text after first tool. Only 2 API calls, 1 tool execution."""
        generator, mock_client = _make_generator()
        mock_client.messages.create.side_effect = [
            make_tool_use_response("search_course_content", {"query": "MCP"}),
            make_text_response("Here is the answer."),
        ]

        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "results"

        result = generator.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        assert result == "Here is the answer."
        assert mock_client.messages.create.call_count == 2
        assert mock_tm.execute_tool.call_count == 1

    def test_tool_execution_error_terminates_rounds(self):
        """execute_tool raises → error sent as tool_result with is_error → follow-up has no tools."""
        generator, mock_client = _make_generator()
        mock_client.messages.create.side_effect = [
            make_tool_use_response(
                "search_course_content", {"query": "MCP"}, tool_use_id="toolu_err"
            ),
            make_text_response("Sorry, I could not retrieve results."),
        ]

        mock_tm = MagicMock()
        mock_tm.execute_tool.side_effect = RuntimeError("Search service unavailable")

        result = generator.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tm,
        )

        assert result == "Sorry, I could not retrieve results."
        # Follow-up call should NOT include tools (error forces last round)
        second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        assert "tools" not in second_call_kwargs

        # Verify error was sent as tool_result with is_error
        messages = second_call_kwargs["messages"]
        tool_result_msg = messages[-1]
        assert tool_result_msg["content"][0]["is_error"] is True
        assert "Search service unavailable" in tool_result_msg["content"][0]["content"]

    def test_message_accumulation_across_rounds(self):
        """Third call has 5 messages: user, asst/tool1, user/result1, asst/tool2, user/result2."""
        generator, mock_client = _make_generator()
        mock_client.messages.create.side_effect = [
            make_tool_use_response(
                "get_course_outline", {"course_name": "AI"}, tool_use_id="toolu_A"
            ),
            make_tool_use_response(
                "search_course_content", {"query": "lesson 4"}, tool_use_id="toolu_B"
            ),
            make_text_response("Final answer"),
        ]

        mock_tm = MagicMock()
        mock_tm.execute_tool.side_effect = ["outline data", "search data"]

        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]
        generator.generate_response(query="test", tools=tools, tool_manager=mock_tm)

        third_call_kwargs = mock_client.messages.create.call_args_list[2].kwargs
        messages = third_call_kwargs["messages"]

        assert len(messages) == 5
        assert messages[0]["role"] == "user"  # original query
        assert messages[1]["role"] == "assistant"  # tool call 1
        assert messages[2]["role"] == "user"  # tool result 1
        assert messages[3]["role"] == "assistant"  # tool call 2
        assert messages[4]["role"] == "user"  # tool result 2

        # Verify correct tool_use_ids
        assert messages[2]["content"][0]["tool_use_id"] == "toolu_A"
        assert messages[4]["content"][0]["tool_use_id"] == "toolu_B"

    def test_max_rounds_enforced(self):
        """Even with 2 tool requests, only 2 executions occur and 3rd call has no tools."""
        generator, mock_client = _make_generator()
        mock_client.messages.create.side_effect = [
            make_tool_use_response(
                "search_course_content", {"query": "q1"}, tool_use_id="toolu_1"
            ),
            make_tool_use_response(
                "search_course_content", {"query": "q2"}, tool_use_id="toolu_2"
            ),
            make_text_response("Done"),
        ]

        mock_tm = MagicMock()
        mock_tm.execute_tool.side_effect = ["r1", "r2"]

        tools = [{"name": "search_course_content"}]
        result = generator.generate_response(
            query="test", tools=tools, tool_manager=mock_tm
        )

        assert result == "Done"
        assert mock_tm.execute_tool.call_count == 2
        assert mock_client.messages.create.call_count == 3

        # 3rd call (final round) must not have tools
        third_call_kwargs = mock_client.messages.create.call_args_list[2].kwargs
        assert "tools" not in third_call_kwargs

    def test_system_prompt_allows_multi_step(self):
        """SYSTEM_PROMPT no longer contains 'One tool call per query maximum'."""
        assert "One tool call per query maximum" not in AIGenerator.SYSTEM_PROMPT
