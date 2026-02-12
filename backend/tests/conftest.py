import sys
import os
import pytest
from unittest.mock import MagicMock

# Ensure backend modules are importable with bare imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager


# --- Pydantic models (mirrored from app.py) ---

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str

class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


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


@pytest.fixture
def mock_rag_system():
    """A MagicMock that behaves like RAGSystem for API-level tests."""
    rag = MagicMock()
    rag.query.return_value = ("This is a test answer.", ["Source A"])
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Intro to MCP", "Advanced RAG"],
    }
    rag.session_manager.create_session.return_value = "session_1"
    rag.session_manager.delete_session.return_value = True
    return rag


@pytest.fixture
def test_app(mock_rag_system):
    """A FastAPI app with the same endpoints as app.py but no static-file mount.

    Uses the mock_rag_system so tests never touch ChromaDB, embeddings, or
    the Anthropic API.
    """
    app = FastAPI()

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        deleted = mock_rag_system.session_manager.delete_session(session_id)
        return {"success": deleted}

    return app


@pytest.fixture
def client(test_app):
    """Starlette/FastAPI TestClient bound to the test app."""
    return TestClient(test_app)
