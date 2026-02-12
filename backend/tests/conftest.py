import sys
import os
import pytest
from unittest.mock import MagicMock

# Ensure backend modules are importable with bare imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager


# --- Fixtures ---

@pytest.fixture
def mock_vector_store():
    """A MagicMock that behaves like VectorStore without ChromaDB/embeddings."""
    store = MagicMock(spec=VectorStore)
    store.max_results = 5
    store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
    store.get_lesson_link.return_value = None
    store.get_course_outline.return_value = None
    return store


@pytest.fixture
def sample_search_results():
    """Pre-built SearchResults with realistic course data."""
    return SearchResults(
        documents=[
            "MCP servers expose tools that AI models can call.",
            "The protocol defines a client-server architecture."
        ],
        metadata=[
            {"course_title": "Introduction to MCP", "lesson_number": 2, "chunk_index": 0},
            {"course_title": "Introduction to MCP", "lesson_number": 3, "chunk_index": 1}
        ],
        distances=[0.3, 0.5]
    )


@pytest.fixture
def error_search_results():
    """SearchResults with an error message."""
    return SearchResults.empty("No course found matching 'Nonexistent'")


@pytest.fixture
def search_tool(mock_vector_store):
    """A CourseSearchTool wired to the mock vector store."""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def tool_manager(mock_vector_store):
    """A ToolManager with search and outline tools registered."""
    tm = ToolManager()
    tm.register_tool(CourseSearchTool(mock_vector_store))
    tm.register_tool(CourseOutlineTool(mock_vector_store))
    return tm
