# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

**Always use `uv` for package management and running the server. Do not use `pip` directly.**

```bash
# Install dependencies
uv sync

# Run the application (from repo root)
./run.sh
# OR manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# Access points
# Web UI: http://localhost:8000
# API docs: http://localhost:8000/docs
```

## Environment Setup

Requires `.env` file in repo root with:
```
ANTHROPIC_API_KEY=your_key_here
```

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot with a FastAPI backend and vanilla JS frontend.

### Query Flow

1. Frontend sends POST to `/api/query` with `{query, session_id}`
2. `RAGSystem.query()` orchestrates the flow
3. `AIGenerator` calls Claude API with tool definitions
4. Claude decides whether to use `search_course_content` tool
5. If tool used: `CourseSearchTool` → `VectorStore` (ChromaDB) → semantic search
6. Claude receives search results, synthesizes final answer
7. Response returns with answer text and source attributions

### Key Backend Components

- **app.py**: FastAPI endpoints, startup document loading
- **rag_system.py**: Central orchestrator connecting all components
- **ai_generator.py**: Claude API integration with tool-use support (two API calls when tools are used)
- **vector_store.py**: ChromaDB wrapper with two collections (`course_catalog` for metadata, `course_content` for chunks)
- **document_processor.py**: Parses course text files, chunks with sentence awareness (800 char chunks, 100 char overlap)
- **search_tools.py**: Tool definitions for Claude's tool-use feature
- **session_manager.py**: Conversation history per session (max 2 exchanges)
- **config.py**: Loads settings from environment/defaults

### Document Format

Course files in `/docs` follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [title]
Lesson Link: [url]
[content...]

Lesson 1: [title]
[content...]
```

### Frontend

Static files served by FastAPI from `/frontend`. Uses `marked.js` for markdown rendering. Session state tracked client-side via `currentSessionId`.

### Data Persistence

ChromaDB stores vectors in `./chroma_db/`. Documents auto-load from `/docs` on startup (skips already-indexed courses).
