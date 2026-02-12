"""Tests for FastAPI endpoints (/api/query, /api/courses, /api/session)."""

import pytest


@pytest.mark.api
class TestQueryEndpoint:

    def test_successful_query(self, client, mock_rag_system):
        """POST /api/query returns answer, sources, and session_id."""
        resp = client.post("/api/query", json={"query": "What is MCP?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "This is a test answer."
        assert data["sources"] == ["Source A"]
        assert data["session_id"] == "session_1"

    def test_query_with_session_id(self, client, mock_rag_system):
        """When session_id is provided it is forwarded, not generated."""
        resp = client.post(
            "/api/query",
            json={"query": "Tell me more", "session_id": "existing_session"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "existing_session"
        mock_rag_system.query.assert_called_once_with("Tell me more", "existing_session")

    def test_query_creates_session_when_missing(self, client, mock_rag_system):
        """When no session_id is sent, one is created via session_manager."""
        resp = client.post("/api/query", json={"query": "hello"})
        assert resp.status_code == 200
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_missing_field(self, client):
        """POST /api/query without 'query' field returns 422."""
        resp = client.post("/api/query", json={})
        assert resp.status_code == 422

    def test_query_empty_string(self, client, mock_rag_system):
        """An empty query string is still valid at the schema level."""
        resp = client.post("/api/query", json={"query": ""})
        assert resp.status_code == 200
        mock_rag_system.query.assert_called_once()

    def test_query_server_error(self, client, mock_rag_system):
        """If rag_system.query raises, the endpoint returns 500."""
        mock_rag_system.query.side_effect = Exception("upstream failure")
        resp = client.post("/api/query", json={"query": "boom"})
        assert resp.status_code == 500
        assert "upstream failure" in resp.json()["detail"]


@pytest.mark.api
class TestCoursesEndpoint:

    def test_get_courses(self, client, mock_rag_system):
        """GET /api/courses returns total_courses and course_titles."""
        resp = client.get("/api/courses")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_courses"] == 2
        assert "Intro to MCP" in data["course_titles"]
        assert "Advanced RAG" in data["course_titles"]

    def test_get_courses_server_error(self, client, mock_rag_system):
        """If get_course_analytics raises, the endpoint returns 500."""
        mock_rag_system.get_course_analytics.side_effect = Exception("db down")
        resp = client.get("/api/courses")
        assert resp.status_code == 500
        assert "db down" in resp.json()["detail"]


@pytest.mark.api
class TestSessionEndpoint:

    def test_delete_session_success(self, client, mock_rag_system):
        """DELETE /api/session/{id} returns success: true for known session."""
        resp = client.delete("/api/session/session_1")
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        mock_rag_system.session_manager.delete_session.assert_called_once_with("session_1")

    def test_delete_session_not_found(self, client, mock_rag_system):
        """DELETE for unknown session returns success: false."""
        mock_rag_system.session_manager.delete_session.return_value = False
        resp = client.delete("/api/session/nonexistent")
        assert resp.status_code == 200
        assert resp.json()["success"] is False
