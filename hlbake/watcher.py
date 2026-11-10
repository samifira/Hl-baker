# SPDX-License-Identifier: GPL-3.0-or-later
"""Watch mesh files and report them once a new version has been completely written."""
import os
import time

from PySide6.QtCore import QObject, QFileSystemWatcher, QTimer, Signal


def _signature(path):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


class MeshFileWatcher(QObject):
    """Emits files_changed([paths]) after a watched file changed and then stayed unchanged for
    `settle` seconds (exporters write big files over several seconds).

    Watches the files *and* their folders: many exporters save to a temporary file and rename it
    over the original, which makes a plain file watch silently stop."""
    files_changed = Signal(list)

    def __init__(self, parent=None, settle=1.0, poll_ms=400):
        super().__init__(parent)
        self.settle = settle
        self.enabled = True
        self._fs = QFileSystemWatcher(self)
        self._fs.fileChanged.connect(self._poke)
        self._fs.directoryChanged.connect(self._poke_dir)
        self._known = {}       # path -> signature of the version currently loaded
        self._pending = {}     # path -> (signature, time first seen with that signature)
        self._timer = QTimer(self, interval=poll_ms, timeout=self._poll)

    def watch(self, paths):
        """Replace the watched set; current file versions count as already loaded."""
        paths = [os.path.abspath(p) for p in paths]
        old_files, old_dirs = self._fs.files(), self._fs.directories()
        if old_files:
            self._fs.removePaths(old_files)
        if old_dirs:
            self._fs.removePaths(old_dirs)
        self._known = {p: _signature(p) for p in paths}
        self._pending.clear()
        existing = [p for p in paths if os.path.exists(p)]
        if existing:
            self._fs.addPaths(existing)
        dirs = sorted({os.path.dirname(p) for p in paths if os.path.isdir(os.path.dirname(p))})
        if dirs:
            self._fs.addPaths(dirs)

    def mark_loaded(self, paths=None):
        """Call after (re)loading: the versions on disk now are the loaded ones."""
        for p in (paths or list(self._known)):
            p = os.path.abspath(p)
            if p in self._known:
                self._known[p] = _signature(p)
                self._pending.pop(p, None)

    def _poke(self, path):
        path = os.path.abspath(path)
        if path in self._known and self.enabled:
            self._pending.setdefault(path, (None, 0.0))
            if not self._timer.isActive():
                self._timer.start()

    def _poke_dir(self, d):
        for p in self._known:
            if os.path.dirname(p) == os.path.abspath(d):
                self._poke(p)

    def _poll(self):
        now = time.monotonic()
        ready = []
        for p, (sig_seen, since) in list(self._pending.items()):
            sig = _signature(p)
            if sig is None:                     # mid-rename / being rewritten: wait
                self._pending[p] = (None, now)
                continue
            if p not in self._fs.files():       # re-arm after an atomic replace
                self._fs.addPath(p)
            if sig != sig_seen:
                self._pending[p] = (sig, now)
                continue
            if now - since >= self.settle:
                del self._pending[p]
                if sig != self._known.get(p) and sig[1] > 0:
                    ready.append(p)
        if not self._pending:
            self._timer.stop()
        if ready:
            self.files_changed.emit(ready)
