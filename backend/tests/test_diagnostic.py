"""Diagnostic tests using REAL components to identify what's broken in the pipeline.

These tests hit the actual ChromaDB and (optionally) the Anthropic API to pinpoint
where the 'query failed' error originates.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import config
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager


# --- VectorStore diagnostics ---

class TestVectorStoreReal:
    """Tests against the real ChromaDB to verify data is loaded and searchable."""

    @pytest.fixture(autouse=True)
    def setup_store(self):
        """Create a real VectorStore pointing at the existing chroma_db."""
        chroma_path = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
        self.store = VectorStore(
            chroma_path=chroma_path,
            embedding_model=config.EMBEDDING_MODEL,
            max_results=5
        )

    def test_chroma_db_has_courses(self):
        """Verify course_catalog collection has data."""
        count = self.store.get_course_count()
        print(f"\n  [DIAG] Course count in catalog: {count}")
        assert count > 0, "ChromaDB course_catalog is EMPTY — no courses loaded"

    def test_chroma_db_has_content(self):
        """Verify course_content collection has data."""
        results = self.store.course_content.get()
        content_count = len(results['ids']) if results and 'ids' in results else 0
        print(f"\n  [DIAG] Content chunks in store: {content_count}")
        assert content_count > 0, "ChromaDB course_content is EMPTY — no chunks indexed"

    def test_search_returns_results(self):
        """Verify a basic semantic search returns documents."""
        results = self.store.search(query="What is computer use?")
        print(f"\n  [DIAG] Search returned {len(results.documents)} documents")
        if results.error:
            print(f"  [DIAG] Search error: {results.error}")
        for i, (doc, meta) in enumerate(zip(results.documents, results.metadata)):
            print(f"  [DIAG] Result {i}: course={meta.get('course_title')}, lesson={meta.get('lesson_number')}, dist={results.distances[i]:.3f}")
            print(f"  [DIAG]   Snippet: {doc[:100]}...")
        assert not results.error, f"Search returned error: {results.error}"
        assert not results.is_empty(), "Search returned ZERO results for 'What is computer use?'"

    def test_course_name_resolution(self):
        """Verify _resolve_course_name finds courses by partial name."""
        titles = self.store.get_existing_course_titles()
        print(f"\n  [DIAG] All course titles: {titles}")

        if titles:
            # Try resolving with a partial name from the first title
            first_title = titles[0]
            partial = first_title.split()[0]  # First word
            resolved = self.store._resolve_course_name(partial)
            print(f"  [DIAG] Partial '{partial}' resolved to: {resolved}")
            assert resolved is not None, f"Failed to resolve partial name '{partial}'"

    def test_search_with_course_filter(self):
        """Verify search works with a course_name filter."""
        titles = self.store.get_existing_course_titles()
        if not titles:
            pytest.skip("No courses in store")

        results = self.store.search(query="introduction", course_name=titles[0])
        print(f"\n  [DIAG] Filtered search for course '{titles[0]}': {len(results.documents)} results")
        if results.error:
            print(f"  [DIAG] Error: {results.error}")
        assert not results.error, f"Filtered search error: {results.error}"


# --- CourseSearchTool with real VectorStore ---

class TestCourseSearchToolReal:
    """Tests CourseSearchTool.execute() with the real VectorStore."""

    @pytest.fixture(autouse=True)
    def setup_tool(self):
        chroma_path = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
        store = VectorStore(
            chroma_path=chroma_path,
            embedding_model=config.EMBEDDING_MODEL,
            max_results=5
        )
        self.tool = CourseSearchTool(store)

    def test_execute_returns_content(self):
        """Verify execute() returns formatted content, not an error."""
        result = self.tool.execute(query="What is computer use with Anthropic?")
        print(f"\n  [DIAG] CourseSearchTool.execute() returned:\n{result[:500]}")
        assert "No relevant content found" not in result, "Search returned no results"
        assert "error" not in result.lower() or "[" in result, f"Search returned error: {result}"

    def test_execute_populates_sources(self):
        """Verify execute() populates last_sources with real data."""
        self.tool.execute(query="What is computer use?")
        print(f"\n  [DIAG] Sources: {self.tool.last_sources}")
        assert len(self.tool.last_sources) > 0, "No sources populated after search"


# --- API key diagnostic ---

class TestAPIKeyConfig:
    """Verify the API key is configured."""

    def test_api_key_is_set(self):
        """Check that ANTHROPIC_API_KEY is not empty."""
        print(f"\n  [DIAG] API key length: {len(config.ANTHROPIC_API_KEY)}")
        print(f"  [DIAG] API key prefix: {config.ANTHROPIC_API_KEY[:10]}..." if config.ANTHROPIC_API_KEY else "  [DIAG] API key is EMPTY")
        assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is empty — AI generator will fail"
        assert config.ANTHROPIC_API_KEY != "your-anthropic-api-key-here", "API key is still the placeholder value"

    def test_api_key_format(self):
        """Check that the API key has a valid format (starts with sk-ant-)."""
        if not config.ANTHROPIC_API_KEY:
            pytest.skip("No API key configured")
        print(f"\n  [DIAG] API key starts with: {config.ANTHROPIC_API_KEY[:7]}")
        assert config.ANTHROPIC_API_KEY.startswith("sk-ant-"), \
            f"API key has unexpected format (starts with '{config.ANTHROPIC_API_KEY[:7]}')"


# --- Full pipeline diagnostic (uses real API) ---

class TestFullPipelineDiagnostic:
    """End-to-end test using real VectorStore + real Anthropic API.

    Marked with a custom marker so it can be skipped if no API key is available.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        if not config.ANTHROPIC_API_KEY:
            pytest.skip("No API key — skipping live API test")

        from rag_system import RAGSystem
        self.rag = RAGSystem(config)

        # Ensure documents are loaded
        docs_path = os.path.join(os.path.dirname(__file__), '..', '..', 'docs')
        if os.path.exists(docs_path):
            self.rag.add_course_folder(docs_path, clear_existing=False)

    def test_full_query_returns_answer(self):
        """The complete RAG pipeline should return a non-empty answer."""
        try:
            response, sources = self.rag.query("What is computer use?")
            print(f"\n  [DIAG] Full pipeline response: {response[:300]}")
            print(f"  [DIAG] Sources: {sources}")
            assert response, "Pipeline returned empty response"
            assert "error" not in response.lower() or len(response) > 50, \
                f"Pipeline returned error-like response: {response}"
        except Exception as e:
            print(f"\n  [DIAG] Full pipeline EXCEPTION: {type(e).__name__}: {e}")
            raise
