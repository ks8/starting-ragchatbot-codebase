"""Tests for CourseSearchTool.execute() and ToolManager."""

from vector_store import SearchResults

# --- CourseSearchTool.execute() tests ---


class TestCourseSearchToolExecute:

    def test_execute_returns_formatted_results(
        self, search_tool, mock_vector_store, sample_search_results
    ):
        """Happy path: returns [Title - Lesson N] header + content blocks joined by double newlines."""
        mock_vector_store.search.return_value = sample_search_results

        result = search_tool.execute(query="What is MCP?")

        assert "[Introduction to MCP - Lesson 2]" in result
        assert "MCP servers expose tools that AI models can call." in result
        assert "[Introduction to MCP - Lesson 3]" in result
        assert "The protocol defines a client-server architecture." in result
        # Blocks separated by double newline
        assert "\n\n" in result

    def test_execute_populates_last_sources(
        self, search_tool, mock_vector_store, sample_search_results
    ):
        """After successful search, tool.last_sources is populated."""
        mock_vector_store.search.return_value = sample_search_results

        search_tool.execute(query="What is MCP?")

        assert len(search_tool.last_sources) == 2
        assert "Introduction to MCP - Lesson 2" in search_tool.last_sources[0]
        assert "Introduction to MCP - Lesson 3" in search_tool.last_sources[1]

    def test_execute_sources_with_lesson_links(
        self, search_tool, mock_vector_store, sample_search_results
    ):
        """When get_lesson_link returns a URL, source is formatted as markdown link."""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson2"

        search_tool.execute(query="What is MCP?")

        # Sources should contain markdown links
        assert any("https://example.com/lesson2" in s for s in search_tool.last_sources)
        assert any(
            "[Introduction to MCP - Lesson" in s for s in search_tool.last_sources
        )

    def test_execute_returns_error_string(
        self, search_tool, mock_vector_store, error_search_results
    ):
        """When results.error is set, returns that error string directly."""
        mock_vector_store.search.return_value = error_search_results

        result = search_tool.execute(query="nonexistent topic")

        assert result == "No course found matching 'Nonexistent'"

    def test_execute_empty_no_filter(self, search_tool, mock_vector_store):
        """Returns 'No relevant content found.' when results are empty and no filters."""
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = search_tool.execute(query="something obscure")

        assert result == "No relevant content found."

    def test_execute_empty_with_course_filter(self, search_tool, mock_vector_store):
        """Returns filter info mentioning course name when empty with course filter."""
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = search_tool.execute(query="something", course_name="MCP")

        assert result == "No relevant content found in course 'MCP'."

    def test_execute_empty_with_lesson_filter(self, search_tool, mock_vector_store):
        """Returns filter info mentioning lesson number when empty with lesson filter."""
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = search_tool.execute(query="something", lesson_number=3)

        assert result == "No relevant content found in lesson 3."

    def test_execute_empty_with_both_filters(self, search_tool, mock_vector_store):
        """Returns combined filter info when empty with both filters."""
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = search_tool.execute(
            query="something", course_name="MCP", lesson_number=3
        )

        assert "in course 'MCP'" in result
        assert "in lesson 3" in result

    def test_execute_passes_params_to_store(self, search_tool, mock_vector_store):
        """Verifies store.search is called with the exact parameters passed to execute."""
        search_tool.execute(
            query="What is MCP?", course_name="MCP Course", lesson_number=2
        )

        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?", course_name="MCP Course", lesson_number=2
        )

    def test_execute_metadata_missing_lesson(self, search_tool, mock_vector_store):
        """Handles metadata with lesson_number=None — header is [Title] without lesson suffix."""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Some course content."],
            metadata=[
                {"course_title": "Test Course", "lesson_number": None, "chunk_index": 0}
            ],
            distances=[0.2],
        )

        result = search_tool.execute(query="test")

        assert "[Test Course]" in result
        assert "Lesson" not in result


# --- ToolManager tests ---


class TestToolManager:

    def test_tool_manager_execute_dispatches(
        self, tool_manager, mock_vector_store, sample_search_results
    ):
        """execute_tool dispatches to the correct tool's execute method."""
        mock_vector_store.search.return_value = sample_search_results

        result = tool_manager.execute_tool(
            "search_course_content", query="What is MCP?"
        )

        assert "[Introduction to MCP" in result

    def test_tool_manager_unknown_tool(self, tool_manager):
        """Returns 'Tool not found' for unregistered tool names."""
        result = tool_manager.execute_tool("nonexistent_tool")

        assert result == "Tool 'nonexistent_tool' not found"

    def test_tool_manager_get_and_reset_sources(
        self, tool_manager, mock_vector_store, sample_search_results
    ):
        """Sources populated after search and cleared after reset."""
        mock_vector_store.search.return_value = sample_search_results

        tool_manager.execute_tool("search_course_content", query="MCP")
        sources = tool_manager.get_last_sources()
        assert len(sources) == 2

        tool_manager.reset_sources()
        assert tool_manager.get_last_sources() == []
