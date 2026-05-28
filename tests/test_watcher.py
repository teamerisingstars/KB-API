import os
import tempfile
import time
from app.watcher import KnowledgeWatcher


def test_watcher_calls_callback_on_md_change():
    called = []
    with tempfile.TemporaryDirectory() as tmpdir:
        watcher = KnowledgeWatcher(tmpdir, lambda: called.append(1), debounce_ms=100)
        watcher.start()
        try:
            with open(os.path.join(tmpdir, "new.md"), "w") as f:
                f.write("## New\n\nContent.\n")
            time.sleep(0.4)
        finally:
            watcher.stop()
    assert len(called) >= 1


def test_watcher_ignores_non_md_files():
    called = []
    with tempfile.TemporaryDirectory() as tmpdir:
        watcher = KnowledgeWatcher(tmpdir, lambda: called.append(1), debounce_ms=100)
        watcher.start()
        try:
            with open(os.path.join(tmpdir, "notes.txt"), "w") as f:
                f.write("not markdown")
            time.sleep(0.4)
        finally:
            watcher.stop()
    assert len(called) == 0


def test_watcher_debounces_rapid_saves():
    called = []
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "test.md")
        watcher = KnowledgeWatcher(tmpdir, lambda: called.append(1), debounce_ms=300)
        watcher.start()
        try:
            for i in range(5):
                with open(md_path, "w") as f:
                    f.write(f"## Section {i}\n\nContent {i}.\n")
                time.sleep(0.05)
            time.sleep(0.6)
        finally:
            watcher.stop()
    # 5 rapid saves should produce 1 or 2 callbacks, not 5
    assert len(called) <= 2
