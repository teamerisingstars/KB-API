import logging
import threading

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class _DebounceHandler(FileSystemEventHandler):
    def __init__(self, callback, debounce_ms: int) -> None:
        super().__init__()
        self._callback = callback
        self._debounce_s = debounce_ms / 1000.0
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def on_any_event(self, event) -> None:
        if event.is_directory:
            return
        if not str(event.src_path).endswith(".md"):
            return
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_s, self._callback)
            self._timer.start()


class KnowledgeWatcher:
    def __init__(self, knowledge_dir: str, callback, debounce_ms: int = 500) -> None:
        self._handler = _DebounceHandler(callback, debounce_ms)
        self._observer = Observer()
        self._observer.schedule(self._handler, knowledge_dir, recursive=True)

    def start(self) -> None:
        self._observer.start()
        logger.info("Knowledge base watcher started.")

    def stop(self) -> None:
        with self._handler._lock:
            if self._handler._timer is not None:
                self._handler._timer.cancel()
                self._handler._timer = None
        self._observer.stop()
        self._observer.join()
        logger.info("Knowledge base watcher stopped.")
