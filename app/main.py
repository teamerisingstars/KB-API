import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.config import KNOWLEDGE_DIR, WATCHER_DEBOUNCE_MS
from app.indexer import IndexStore, build_index
from app.models import AskRequest, AskResponse, HealthResponse, ReloadResponse, SectionInfo, SectionsResponse
from app.searcher import search
from app.watcher import KnowledgeWatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_index_store = IndexStore()
_watcher: KnowledgeWatcher | None = None


def _trigger_reindex() -> None:
    threading.Thread(
        target=build_index, args=(KNOWLEDGE_DIR, _index_store), daemon=True
    ).start()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _watcher
    if not Path(KNOWLEDGE_DIR).exists():
        logger.error("Knowledge directory not found: %s — API will return no matches.", KNOWLEDGE_DIR)
    else:
        build_index(KNOWLEDGE_DIR, _index_store)
        _watcher = KnowledgeWatcher(KNOWLEDGE_DIR, _trigger_reindex, WATCHER_DEBOUNCE_MS)
        _watcher.start()
    yield
    if _watcher:
        _watcher.stop()


app = FastAPI(title="Knowledge Base API", version="1.0.0", lifespan=_lifespan)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return search(request.question, _index_store)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        indexed_sections=len(_index_store.sections),
        indexed_files=_index_store.file_count,
        last_indexed=_index_store.last_indexed,
    )


@app.post("/reload", response_model=ReloadResponse)
def reload() -> ReloadResponse:
    _trigger_reindex()
    return ReloadResponse(status="reindexing", message="Reindex triggered.")


@app.get("/sections", response_model=SectionsResponse)
def sections() -> SectionsResponse:
    return SectionsResponse(
        sections=[
            SectionInfo(heading=s.heading, source=s.source_file)
            for s in _index_store.sections
        ]
    )
