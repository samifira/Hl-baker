# SPDX-License-Identifier: GPL-3.0-or-later
"""HL Baker desktop UI (PySide6). The UI never imports bpy: all Blender work runs in a
separate worker process (this same executable with the 'worker' argument)."""
import copy
import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import fields

import numpy as np

from PySide6.QtCore import Qt, QProcess, QObject, Signal, QTimer, QSize, QSettings, QUrl
from PySide6.QtGui import (QAction, QColor, QIcon, QKeySequence, QPalette, QPixmap, QImage,
                           QPainter, QDesktopServices, QFont)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QGroupBox, QPushButton, QToolButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QTreeWidget, QTreeWidgetItem,
    QTabWidget, QPlainTextEdit, QProgressBar, QFileDialog, QMessageBox, QScrollArea, QSplitter,
    QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene, QSlider, QMenu, QAbstractItemView,
    QSizePolicy, QFrame, QDialog, QDialogButtonBox)

from .config import (Project, FileEntry, GroupConfig, Settings, MAPS, MAP_ORDER, MATERIAL_MAPS,
                     GAME_PRESET, FORMATS, FORMAT_LABELS, MESH_EXTS, PROJECT_EXT, APP_NAME,
                     APP_VERSION, guess_file_role, enabled_maps)
from .cli import EVENT_PREFIX
from .viewport import ViewerPanel, request_gl_format
from .transform_panel import TransformPanel
from .watcher import MeshFileWatcher
from . import xform, align
from .config import texture_set_name

ROLE_LABELS = [("auto", "Auto (by name)"), ("low", "Low poly"), ("high", "High poly"), ("cage", "Cage")]
SIZES = [256, 512, 1024, 2048, 4096, 8192]


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------
def worker_program():
    if getattr(sys, "frozen", False):
        # use the console sibling executable 'hlbake' (reliable stdout pipes on every OS)
        cli = os.path.join(os.path.dirname(sys.executable),
                           "hlbake.exe" if sys.platform == "win32" else "hlbake")
        return (cli if os.path.exists(cli) else sys.executable), ["worker"]
    main = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hlbake_main.py")
    return sys.executable, [main, "worker"]


class Worker(QObject):
    event = Signal(dict)
    finished = Signal(str, int)   # command, exit code
    output = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc = None
        self.command = None
        self._buf = b""
        self._got_terminal = False

    def busy(self):
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def start(self, command, project, extra=()):
        if self.busy():
            return False
        fd, path = tempfile.mkstemp(suffix=PROJECT_EXT)
        os.close(fd)
        project.save(path)
        prog, args = worker_program()
        self.command = command
        self._buf = b""
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        env = self.proc.processEnvironment()
        env = env.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        if sys.platform == "win32" and hasattr(self.proc, "setCreateProcessArgumentsModifier"):
            def _no_window(args):          # CREATE_NO_WINDOW: keep the worker's console hidden
                args.flags |= 0x08000000
            self.proc.setCreateProcessArgumentsModifier(_no_window)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(lambda code, _st, p=path: self._done(code, p))
        self.proc.start(prog, args + [command, path, *extra])
        return True

    def cancel(self):
        if self.busy():
            self.proc.kill()

    def _read(self):
        self._buf += bytes(self.proc.readAllStandardOutput())
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            text = line.decode("utf-8", "replace").rstrip("\r")
            if text.startswith(EVENT_PREFIX):
                try:
                    self.event.emit(json.loads(text[len(EVENT_PREFIX):]))
                except json.JSONDecodeError:
                    self.output.emit(text)
            elif text.strip() and not text.startswith(("Fra:", "Info: Baking map saved")):
                self.output.emit(text)

    def _done(self, code, path):
        self._read()
        try:
            os.remove(path)
        except OSError:
            pass
        cmd = self.command
        self.proc = None
        self.finished.emit(cmd, code)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------
class ImageView(QGraphicsView):
    """Zoom (wheel) / pan (drag) image viewer."""

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(28, 29, 32))
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self._item = None
        self._fit = True

    def set_image(self, path):
        self.scene().clear()
        self._item = None
        if path and os.path.exists(path):
            pm = QPixmap(path)
            if not pm.isNull():
                self._item = self.scene().addPixmap(pm)
                self.scene().setSceneRect(self._item.boundingRect())
                self._fit = True
                self.fit()

    def fit(self):
        if self._item is not None:
            self.fitInView(self._item, Qt.KeepAspectRatio)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit:
            self.fit()

    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self._fit = False
        self.scale(f, f)

    def mouseDoubleClickEvent(self, e):
        self._fit = True
        self.resetTransform()
        self.fit()


def section(title):
    box = QGroupBox(title)
    box.setObjectName("section")
    return box


def dspin(lo, hi, dec=4, step=0.01, suffix=" m"):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setSuffix(suffix)
    w.setKeyboardTracking(False)
    return w


def ispin(lo, hi, suffix=""):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setSuffix(suffix)
    w.setKeyboardTracking(False)
    return w


def combo(items):
    w = QComboBox()
    for data, label in items:
        w.addItem(label, data)
    return w


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = Project()
        self.project_path = None
        self.objects = []            # last scan result
        self.preview_dir = tempfile.mkdtemp(prefix="hlbake_prev_")
        self.render_path = os.path.join(self.preview_dir, "render.png")
        self.mesh_path = os.path.join(self.preview_dir, "meshes.npz")
        self.map_previews = {}       # label -> png
        self._loading = False
        self.qs = QSettings("HLBaker", "HLBaker")

        self.worker = Worker(self)
        self.worker.event.connect(self.on_event)
        self.worker.output.connect(lambda t: self.log(t, dim=True))
        self.worker.finished.connect(self.on_worker_finished)
        self._pending_render = False
        # transforms / outliner / parity / reload state
        self.scene_pts = {}          # object name -> (role, file, raw points, is_triangle_soup)
        self.obj_file = {}           # object name -> file path
        self.target = None           # ("file", path) | ("object", name) | None
        self.undo_stack, self.redo_stack = [], []
        self._last_edit = 0.0
        self._xf_state = self._transform_state()
        self.parity = {}             # group name -> align.check() result
        self.parity_ignored = set()
        self._parity_gen = 0
        self.maps_present = False
        self._reload_queue = set()
        self._reloading = None
        self.watcher = MeshFileWatcher(self)
        self.watcher.files_changed.connect(self.on_files_changed)
        self.parity_done = _Relay(self)
        self.parity_done.fire.connect(self._parity_results)

        self.setWindowTitle(APP_NAME)
        self.setAcceptDrops(True)
        self.resize(1440, 900)
        self._build_menu()
        self._build_ui()
        self.load_project_into_ui()
        self.scan_timer = QTimer(self, singleShot=True, interval=300, timeout=self.scan)
        self._xf_settle = QTimer(self, singleShot=True, interval=500,
                                 timeout=lambda: (self.refresh_groups_tree(), self.schedule_parity()))

    # ------------------------------------------------------------------ UI
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for text, key, fn in (("&New Project", QKeySequence.New, self.new_project),
                              ("&Open Project…", QKeySequence.Open, self.open_project),
                              ("&Save Project", QKeySequence.Save, self.save_project),
                              ("Save Project &As…", QKeySequence.SaveAs, self.save_project_as)):
            a = QAction(text, self)
            a.setShortcut(key)
            a.triggered.connect(fn)
            m.addAction(a)
        m.addSeparator()
        a = QAction("Add Mesh Files…", self)
        a.setShortcut("Ctrl+I")
        a.triggered.connect(lambda: self.add_files_dialog("auto"))
        m.addAction(a)
        a = QAction("Reload Meshes", self)
        a.setShortcut("F5")
        a.triggered.connect(lambda: self.reload_meshes())
        m.addAction(a)
        a = QAction("Export Corrected Meshes…", self)
        a.triggered.connect(self.export_dialog)
        m.addAction(a)
        m.addSeparator()
        a = QAction("&Quit", self)
        a.setShortcut(QKeySequence.Quit)
        a.triggered.connect(self.close)
        m.addAction(a)
        e = self.menuBar().addMenu("&Edit")
        self.undo_act = QAction("Undo Transform", self)
        self.undo_act.setShortcut(QKeySequence.Undo)
        self.undo_act.triggered.connect(self.undo)
        self.redo_act = QAction("Redo Transform", self)
        self.redo_act.setShortcuts([QKeySequence.Redo, QKeySequence("Ctrl+Shift+Z")])
        self.redo_act.triggered.connect(self.redo)
        e.addAction(self.undo_act)
        e.addAction(self.redo_act)
        h = self.menuBar().addMenu("&Help")
        a = QAction("Naming Conventions", self)
        a.triggered.connect(self.show_naming_help)
        h.addAction(a)
        a = QAction("About", self)
        a.triggered.connect(self.show_about)
        h.addAction(a)

    def _build_ui(self):
        self.w = {}   # settings field -> widget
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(8, 8, 8, 8)
        lv.setSpacing(10)
        lv.addWidget(self._build_files())
        lv.addWidget(self._build_groups())
        lv.addWidget(self._build_transform())
        lv.addWidget(self._build_maps())
        lv.addWidget(self._build_output())
        lv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(470)

        # bake bar
        bar = QWidget()
        bl = QVBoxLayout(bar)
        bl.setContentsMargins(8, 4, 8, 8)
        row = QHBoxLayout()
        self.bake_btn = QPushButton("Bake")
        self.bake_btn.setObjectName("bake")
        self.bake_btn.setMinimumHeight(42)
        self.bake_btn.clicked.connect(self.bake)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(42)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        row.addWidget(self.bake_btn, 3)
        row.addWidget(self.cancel_btn, 1)
        bl.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("Ready")
        self.progress.setValue(0)
        bl.addWidget(self.progress)

        leftcol = QWidget()
        lc = QVBoxLayout(leftcol)
        lc.setContentsMargins(0, 0, 0, 0)
        lc.setSpacing(0)
        lc.addWidget(scroll, 1)
        lc.addWidget(bar)

        # right side
        self.tabs = QTabWidget()
        self.viewer3d = ViewerPanel()
        self.tabs.addTab(self.viewer3d, "3D Viewer")
        self.tabs.addTab(self._build_maps_view(), "Baked Maps")
        self.tabs.addTab(self._build_render_view(), "Cycles Render")
        self.logbox = QPlainTextEdit(readOnly=True)
        self.logbox.setFont(QFont("Monospace", 9))
        self.logbox.setMaximumBlockCount(5000)
        self.tabs.addTab(self.logbox, "Log")

        split = QSplitter()
        split.addWidget(leftcol)
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([500, 940])
        self.setCentralWidget(split)
        self.statusBar().showMessage("Drop mesh files (FBX, OBJ, glTF…) anywhere to start")
        self.stale_label = QLabel("")
        self.stale_label.setObjectName("stale")
        self.stale_label.setVisible(False)
        self.statusBar().addPermanentWidget(self.stale_label)

    # -- files
    def _build_files(self):
        box = section("1  Meshes")
        v = QVBoxLayout(box)
        self.files_table = QTableWidget(0, 2)
        self.files_table.setHorizontalHeaderLabels(["File", "Role"])
        self.files_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.files_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.files_table.setColumnWidth(1, 150)
        self.files_table.verticalHeader().setVisible(False)
        self.files_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.files_table.setMinimumHeight(110)
        self.files_table.setMaximumHeight(160)
        self.files_table.itemSelectionChanged.connect(self.file_selected)
        v.addWidget(self.files_table)
        row = QHBoxLayout()
        for text, role in (("+ Low…", "low"), ("+ High…", "high"), ("+ Cage…", "cage"),
                           ("+ Mixed…", "auto")):
            b = QPushButton(text)
            b.setToolTip({"low": "Every mesh in these files is a low poly",
                          "high": "Every mesh in these files is a high poly",
                          "cage": "Cage meshes (same topology as their low poly)",
                          "auto": "Files with meshes named *_low / *_high / *_cage"}[role])
            b.clicked.connect(lambda _=False, r=role: self.add_files_dialog(r))
            row.addWidget(b)
        rm = QPushButton("Remove")
        rm.clicked.connect(self.remove_files)
        row.addWidget(rm)
        v.addLayout(row)
        row2 = QHBoxLayout()
        self.auto_reload = QCheckBox("Auto-reload changed files")
        self.auto_reload.setToolTip("Reload a mesh file as soon as a new version is saved "
                                    "(never re-bakes automatically)")
        self.auto_reload.setChecked(self.qs.value("auto_reload", True, type=bool))
        self.auto_reload.toggled.connect(self._auto_reload_toggled)
        row2.addWidget(self.auto_reload)
        row2.addStretch(1)
        rl = QPushButton("Reload")
        rl.setToolTip("Reload all mesh files now (F5)")
        rl.clicked.connect(lambda: self.reload_meshes())
        row2.addWidget(rl)
        v.addLayout(row2)
        return box

    # -- groups
    def _build_groups(self):
        box = section("2  Bake Groups")
        v = QVBoxLayout(box)
        self.groups_tree = QTreeWidget()
        self.groups_tree.setHeaderLabels(["Group / mesh", "", "Cage offset", "Max ray"])
        self.groups_tree.headerItem().setToolTip(1, "Visible in the 3D viewer (Ctrl-click: show only this)")
        self.groups_tree.headerItem().setIcon(1, eye_icon("on"))
        hdr = self.groups_tree.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Fixed)
        hdr.resizeSection(1, 30)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.groups_tree.itemClicked.connect(self.tree_clicked)
        self.groups_tree.setMinimumHeight(170)
        self.groups_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.groups_tree.customContextMenuRequested.connect(self.groups_menu)
        self.groups_tree.itemChanged.connect(self.group_item_changed)
        self.groups_tree.currentItemChanged.connect(self.group_selected)
        v.addWidget(self.groups_tree)
        f = QGridLayout()
        self.g_offset = dspin(0, 1000, 4, 0.005)
        self.g_offset.setToolTip("Cage offset: rays start this far outside the low poly")
        self.g_maxray = dspin(0, 1000, 4, 0.01)
        self.g_maxray.setToolTip("Max ray distance (0 = unlimited)")
        self.g_offset.valueChanged.connect(lambda v: self.set_group_value("cage_offset", v))
        self.g_maxray.valueChanged.connect(lambda v: self.set_group_value("max_ray", v))
        f.addWidget(QLabel("Cage offset"), 0, 0)
        f.addWidget(self.g_offset, 0, 1)
        f.addWidget(QLabel("Max ray"), 0, 2)
        f.addWidget(self.g_maxray, 0, 3)
        all_btn = QPushButton("Apply to all groups")
        all_btn.clicked.connect(self.apply_offsets_all)
        rescan = QPushButton("Re-detect groups")
        rescan.setToolTip("Rebuild groups from mesh names (discards manual changes)")
        rescan.clicked.connect(lambda: self.scan(reset=True))
        f.addWidget(all_btn, 1, 0, 1, 2)
        f.addWidget(rescan, 1, 2, 1, 2)
        v.addLayout(f)
        self.groups_hint = QLabel("")
        self.groups_hint.setWordWrap(True)
        self.groups_hint.setObjectName("hint")
        v.addWidget(self.groups_hint)
        self.parity_box = QWidget()
        self.parity_layout = QVBoxLayout(self.parity_box)
        self.parity_layout.setContentsMargins(0, 0, 0, 0)
        self.parity_layout.setSpacing(4)
        self.parity_box.setVisible(False)
        v.addWidget(self.parity_box)
        return box

    def _build_transform(self):
        box = section("3  Transform")
        v = QVBoxLayout(box)
        self.xf = TransformPanel()
        self.xf.changed.connect(self.on_transform_edited)
        self.xf.reset_requested.connect(self.reset_transform)
        self.xf.align_requested.connect(lambda: self.auto_align_group(self._target_group()))
        self.xf.export_requested.connect(self.export_dialog)
        self.xf.undo_requested.connect(self.undo)
        self.xf.redo_requested.connect(self.redo)
        self.xf.set_undo_state(False, False)
        v.addWidget(self.xf)
        return box

    # -- maps
    def _build_maps(self):
        box = section("4  Maps")
        v = QVBoxLayout(box)
        row = QHBoxLayout()
        for label, keys in (("None", ()), ("Game set", GAME_PRESET), ("All", MAP_ORDER)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, k=keys: self.set_maps(k))
            row.addWidget(b)
        v.addLayout(row)
        grid = QGridLayout()
        geo = [k for k in MAP_ORDER if k not in MATERIAL_MAPS]
        for i, k in enumerate(geo):
            cb = QCheckBox(MAPS[k][0])
            self.w["map_" + k] = cb
            grid.addWidget(cb, i // 2, i % 2)
        v.addLayout(grid)
        lab = QLabel("From high poly materials")
        lab.setObjectName("sub")
        v.addWidget(lab)
        grid2 = QGridLayout()
        for i, k in enumerate(MATERIAL_MAPS):
            cb = QCheckBox(MAPS[k][0])
            self.w["map_" + k] = cb
            grid2.addWidget(cb, i // 3, i % 3)
        v.addLayout(grid2)

        opts = QGroupBox("Map options")
        opts.setObjectName("inner")
        form = QFormLayout(opts)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.opt_rows = {}

        def add(map_key, label, field, widget):
            self.w[field] = widget
            form.addRow(label, widget)
            self.opt_rows.setdefault(map_key, []).append((form.labelForField(widget), widget))

        add("normal", "Normal Y", "normal_flip_y",
            combo([("OPENGL", "OpenGL Y+ (Blender, Unity, Toolbag)"), ("DIRECTX", "DirectX Y- (Unreal, Max)")]))
        add("ao", "AO distance", "ao_distance", dspin(0.0001, 1000, 4, 0.05))
        add("ao", "AO ignore groups", "ao_ignore_groups", QCheckBox("all high polys occlude each other"))
        add("ao", "AO self only", "ao_only_self", QCheckBox("each mesh only occludes itself"))
        add("curvature", "Curvature method", "curv_method",
            combo([("RAY", "Ray traced"), ("POINTINESS", "Vertex (pointiness)")]))
        add("curvature", "Curvature radius", "curv_radius", dspin(0.00001, 100, 4, 0.002))
        add("curvature", "Curvature strength", "curv_strength", dspin(0, 50, 2, 0.25, ""))
        add("thickness", "Thickness distance", "thick_distance", dspin(0.0001, 1000, 4, 0.05))
        add("height", "Height range", "height_auto", QCheckBox("auto"))
        add("height", "Height inner", "height_inner", dspin(0, 1000, 4, 0.01))
        add("height", "Height outer", "height_outer", dspin(0, 1000, 4, 0.01))
        add("position", "Position space", "position_space",
            combo([("BOUNDS", "Normalized to bounds"), ("WORLD", "World (raw, use EXR)")]))
        add("material_id", "Material ID colors", "matid_source",
            combo([("HASH", "Generated"), ("VIEWPORT", "Material viewport color")]))
        v.addWidget(opts)
        for k in MAP_ORDER:
            self.w["map_" + k].toggled.connect(self.update_option_visibility)
        self.w["normal_flip_y"].currentIndexChanged.connect(
            lambda _i: self.viewer3d.set_directx(self.w["normal_flip_y"].currentData() == "DIRECTX"))
        self.w["height_auto"].toggled.connect(self.update_option_visibility)
        self.w["curv_method"].currentIndexChanged.connect(self.update_option_visibility)
        self.opts_box = opts
        return box

    # -- output
    def _build_output(self):
        box = section("5  Output")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        row = QHBoxLayout()
        self.w["output_dir"] = QLineEdit()
        b = QToolButton(text="…")
        b.clicked.connect(self.pick_output_dir)
        o = QToolButton(text="Open")
        o.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(self.w["output_dir"].text())))
        row.addWidget(self.w["output_dir"])
        row.addWidget(b)
        row.addWidget(o)
        form.addRow("Folder", row)
        self.w["base_name"] = QLineEdit()
        form.addRow("Name", self.w["base_name"])
        self.w["output_mode"] = combo([("SINGLE", "Single texture set ({name}_{map})"),
                                       ("PER_GROUP", "Per group ({group}_{map})")])
        form.addRow("Textures", self.w["output_mode"])
        self.w["file_format"] = combo([(k, FORMAT_LABELS[k]) for k in FORMATS])
        form.addRow("Format", self.w["file_format"])
        sz = QHBoxLayout()
        self.w["width"] = combo([(s, str(s)) for s in SIZES])
        self.w["height"] = combo([(s, str(s)) for s in SIZES])
        self.link_size = QCheckBox("square")
        self.link_size.setChecked(True)
        self.w["width"].currentIndexChanged.connect(
            lambda i: self.link_size.isChecked() and self.w["height"].setCurrentIndex(i))
        sz.addWidget(self.w["width"])
        sz.addWidget(QLabel("×"))
        sz.addWidget(self.w["height"])
        sz.addWidget(self.link_size)
        form.addRow("Resolution", sz)
        self.w["supersample"] = combo([("1", "1x (none)"), ("2", "4x"), ("4", "16x (slow)")])
        form.addRow("Anti-aliasing", self.w["supersample"])
        self.w["ray_samples"] = ispin(1, 4096)
        self.w["ray_samples"].setToolTip("Samples for AO, thickness and curvature")
        form.addRow("Ray samples", self.w["ray_samples"])
        pad = QHBoxLayout()
        self.w["padding"] = ispin(0, 1024, " px")
        self.w["padding_full"] = QCheckBox("fill whole texture")
        self.w["padding_full"].toggled.connect(lambda c: self.w["padding"].setEnabled(not c))
        pad.addWidget(self.w["padding"])
        pad.addWidget(self.w["padding_full"])
        form.addRow("Padding", pad)
        self.w["device"] = combo([("GPU", "GPU (falls back to CPU)"), ("CPU", "CPU")])
        form.addRow("Device", self.w["device"])
        self.w["use_uv"] = QComboBox()
        self.w["use_uv"].setEditable(True)
        self.w["use_uv"].lineEdit().setPlaceholderText("active UV map")
        form.addRow("UV map", self.w["use_uv"])
        return box

    # -- right side
    def _build_maps_view(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        self.viewer = ImageView()
        self.thumbs = QListWidget()
        self.thumbs.setViewMode(QListWidget.IconMode)
        self.thumbs.setIconSize(QSize(112, 112))
        self.thumbs.setFlow(QListWidget.LeftToRight)
        self.thumbs.setWrapping(False)
        self.thumbs.setFixedHeight(160)
        self.thumbs.setMovement(QListWidget.Static)
        self.thumbs.setSpacing(4)
        self.thumbs.currentItemChanged.connect(
            lambda it, _p: it and self.viewer.set_image(it.data(Qt.UserRole)))
        self.thumbs.itemDoubleClicked.connect(
            lambda it: QDesktopServices.openUrl(QUrl.fromLocalFile(it.data(Qt.UserRole + 1))))
        self.viewer_hint = QLabel("Baked maps appear here.  Scroll to zoom, drag to pan, "
                                  "double-click to fit.  Double-click a thumbnail to open the file.")
        self.viewer_hint.setObjectName("hint")
        v.addWidget(self.viewer, 1)
        v.addWidget(self.viewer_hint)
        v.addWidget(self.thumbs)
        return w

    def _build_render_view(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        self.render_view = ImageView()
        v.addWidget(self.render_view, 1)
        row = QHBoxLayout()
        self.yaw = QSlider(Qt.Horizontal)
        self.yaw.setRange(-180, 180)
        self.yaw.setValue(35)
        self.pitch = QSlider(Qt.Horizontal)
        self.pitch.setRange(-80, 80)
        self.pitch.setValue(25)
        self.r_samples = ispin(1, 1024)
        self.r_samples.setValue(32)
        self.r_show = combo([("low", "Low poly + baked maps"), ("high", "High poly")])
        self.render_btn = QPushButton("Render")
        self.render_btn.clicked.connect(self.render_preview)
        self.auto_render = QCheckBox("after bake")
        self.auto_render.setChecked(False)
        for lab, wd in (("Orbit", self.yaw), ("Tilt", self.pitch), ("Samples", self.r_samples),
                        ("", self.r_show)):
            if lab:
                row.addWidget(QLabel(lab))
            row.addWidget(wd)
        row.addWidget(self.render_btn)
        row.addWidget(self.auto_render)
        v.addLayout(row)
        for s in (self.yaw, self.pitch):
            s.sliderReleased.connect(self.render_preview)
        return w

    # ------------------------------------------------------------------ settings binding
    def _set_widget(self, w, v):
        if isinstance(w, QCheckBox):
            w.setChecked(bool(v))
        elif isinstance(w, QDoubleSpinBox):
            w.setValue(float(v))
        elif isinstance(w, QSpinBox):
            w.setValue(int(v))
        elif isinstance(w, QComboBox) and w.isEditable():
            w.setEditText(str(v))
        elif isinstance(w, QComboBox):
            i = w.findData(v)
            if i < 0 and isinstance(v, int):           # custom size
                w.addItem(str(v), v)
                i = w.count() - 1
            w.setCurrentIndex(max(i, 0))
        elif isinstance(w, QLineEdit):
            w.setText(str(v))

    def _get_widget(self, w, current):
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QDoubleSpinBox):
            return w.value()
        if isinstance(w, QSpinBox):
            return w.value()
        if isinstance(w, QComboBox) and w.isEditable():
            return w.currentText().strip()
        if isinstance(w, QComboBox):
            return w.currentData()
        if isinstance(w, QLineEdit):
            return w.text().strip()
        return current

    def load_project_into_ui(self):
        self._loading = True
        s = self.project.settings
        if self.link_size is not None:
            self.link_size.setChecked(s.width == s.height)
        for f in fields(Settings):
            if f.name in self.w:
                self._set_widget(self.w[f.name], getattr(s, f.name))
        self.g_offset.setValue(self.project.default_cage_offset)
        self.g_maxray.setValue(self.project.default_max_ray)
        self.refresh_files_table()
        self.refresh_groups_tree()
        self.update_option_visibility()
        self.viewer3d.set_directx(s.normal_flip_y == "DIRECTX")
        self._loading = False
        self.update_title()

    def collect_settings(self):
        s = self.project.settings
        for f in fields(Settings):
            if f.name in self.w:
                setattr(s, f.name, self._get_widget(self.w[f.name], getattr(s, f.name)))
        if not s.output_dir and self.project_path:
            s.output_dir = os.path.join(os.path.dirname(self.project_path), "bakes")
        return s

    def update_option_visibility(self):
        for mk, rows in self.opt_rows.items():
            on = self.w["map_" + mk].isChecked()
            for lab, wd in rows:
                vis = on
                if wd is self.w.get("height_inner") or wd is self.w.get("height_outer"):
                    vis = on and not self.w["height_auto"].isChecked()
                if wd is self.w.get("curv_radius"):
                    vis = on and self.w["curv_method"].currentData() == "RAY"
                lab.setVisible(vis)
                wd.setVisible(vis)
        self.opts_box.setVisible(any(self.w["map_" + k].isChecked() for k in self.opt_rows))

    def set_maps(self, keys):
        for k in MAP_ORDER:
            self.w["map_" + k].setChecked(k in keys)

    # ------------------------------------------------------------------ files
    def refresh_files_table(self):
        t = self.files_table
        t.setRowCount(0)
        for i, fe in enumerate(self.project.files):
            t.insertRow(i)
            it = QTableWidgetItem(os.path.basename(fe.path))
            it.setToolTip(fe.path)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            t.setItem(i, 0, it)
            cb = combo(ROLE_LABELS)
            cb.setCurrentIndex(cb.findData(fe.role))
            cb.currentIndexChanged.connect(lambda _i, c=cb, e=fe: self.set_file_role(e, c.currentData()))
            t.setCellWidget(i, 1, cb)

    def set_file_role(self, fe, role):
        fe.role = role
        self.scan(reset=True)

    def add_files(self, paths, role=None):
        added = 0
        have = {os.path.normcase(f.path) for f in self.project.files}
        for p in paths:
            p = os.path.abspath(p)
            if p.lower().endswith(PROJECT_EXT):
                self.open_project(p)
                return
            if not p.lower().endswith(MESH_EXTS) or os.path.normcase(p) in have:
                continue
            self.project.files.append(FileEntry(p, role or guess_file_role(p)))
            added += 1
        if added:
            if not self.w["output_dir"].text():
                self.w["output_dir"].setText(os.path.join(os.path.dirname(os.path.abspath(paths[0])), "bakes"))
            if self.w["base_name"].text() in ("", "bake"):
                stem = os.path.splitext(os.path.basename(paths[0]))[0]
                from .config import split_name
                self.w["base_name"].setText(split_name(stem)[0])
            self.refresh_files_table()
            self.scan(reset=True)

    def add_files_dialog(self, role):
        last = self.qs.value("last_dir", os.path.expanduser("~"))
        pat = "Meshes (" + " ".join("*" + e for e in MESH_EXTS) + ")"
        paths, _ = QFileDialog.getOpenFileNames(self, "Add mesh files", last, pat + ";;All files (*)")
        if paths:
            self.qs.setValue("last_dir", os.path.dirname(paths[0]))
            self.add_files(paths, role)

    def remove_files(self):
        rows = sorted({i.row() for i in self.files_table.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.project.files[r]
        if rows:
            self.refresh_files_table()
            self.scan(reset=True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        files = []
        for p in paths:
            if os.path.isdir(p):
                for n in sorted(os.listdir(p)):
                    if n.lower().endswith(MESH_EXTS):
                        files.append(os.path.join(p, n))
            else:
                files.append(p)
        self.add_files(files)

    # ------------------------------------------------------------------ scan / groups
    def scan(self, reset=False):
        if not self.project.files:
            self.project.groups = []
            self.objects = []
            self.refresh_groups_tree()
            return
        if self.worker.busy():
            self.scan_timer.start(500)
            return
        self._scan_reset = reset
        self._old_groups = {g.name: g for g in self.project.groups}
        if reset:
            self._restore_groups = None
            self.parity_ignored.clear()
        self.project.groups = [] if reset else self.project.groups
        self.collect_settings()
        self.progress.setRange(0, 0)
        self.progress.setFormat("Loading meshes…")
        self.set_busy(True)
        if reset:
            self.viewer3d.view.clear_textures()
        self.worker.start("scan", self.project, ["--mesh-out", self.mesh_path])

    def apply_scan(self, ev):
        self.objects = ev.get("objects", [])
        self.obj_file = {o["name"]: o["file"] for o in self.objects}
        keep_camera = bool(self._reloading)
        if ev.get("mesh_file"):
            self._load_scene_points(ev["mesh_file"])
            self.viewer3d.view.load_meshes(ev["mesh_file"], keep_camera=keep_camera)
        if ev.get("mesh_error"):
            self.log("3D viewer: could not export meshes: " + ev["mesh_error"])
        restore = getattr(self, "_restore_groups", None)
        unmatched = ev.get("unmatched", [])
        if restore is not None:              # open project / reload: keep the user's groups
            self.project.groups = restore
            self._restore_groups = None
            unmatched = []                    # missing objects are reported separately
        else:
            groups = [GroupConfig(**g) for g in ev.get("groups", [])]
            for g in groups:   # keep per-group tweaks across rescans
                old = self._old_groups.get(g.name)
                if old:
                    g.cage_offset, g.max_ray, g.enabled = old.cage_offset, old.max_ray, old.enabled
                else:
                    g.cage_offset, g.max_ray = self.g_offset.value(), self.g_maxray.value()
            self.project.groups = groups
        names = set(self.obj_file)
        self.project.hidden = [n for n in self.project.hidden if n in names]
        uvs = sorted({u for o in self.objects if o["role"] == "low" for u in o["uvs"]})
        cur = self.w["use_uv"].currentText()
        self.w["use_uv"].clear()
        self.w["use_uv"].addItems([""] + uvs)
        self.w["use_uv"].setEditText(cur)
        self.refresh_groups_tree()
        self.sync_viewer()
        self.update_hints(unmatched)
        n_low = sum(o["role"] == "low" for o in self.objects)
        n_high = sum(o["role"] == "high" for o in self.objects)
        tris = sum(o["tris"] for o in self.objects if o["role"] == "high")
        self.statusBar().showMessage(f"{n_low} low / {n_high} high meshes ({tris:,} high tris) "
                                     f"in {len(self.project.groups)} group(s)")
        self.watcher.watch([fe.path for fe in self.project.files])
        self.schedule_parity()

    def update_hints(self, unmatched=()):
        hints = []
        if unmatched:
            hints.append("No high poly for: " + ", ".join(unmatched))
        names = set(self.obj_file)
        missing = sorted({n for g in self.project.groups for n in [g.low, *g.highs, g.cage]
                          if n and n not in names})
        if missing:
            hints.append("Not found in the loaded files (renamed or deleted?): " + ", ".join(missing))
        used = {n for g in self.project.groups for n in [g.low, *g.highs, g.cage] if n}
        new = [o["name"] for o in self.objects if o["role"] in ("low", "high", "cage")
               and o["name"] not in used]
        if new and self.project.groups:
            hints.append("Not in any group: " + ", ".join(new[:8]) + ("…" if len(new) > 8 else "")
                         + "  (Re-detect groups to include them)")
        unassigned = [o["name"] for o in self.objects if o["role"] == "unassigned"]
        if unassigned:
            hints.append("Ignored (no _low/_high suffix): " + ", ".join(unassigned[:8])
                         + ("…" if len(unassigned) > 8 else ""))
        nouv = [o["name"] for o in self.objects if o["role"] == "low" and not o["uvs"]]
        if nouv:
            hints.append("Low polys without UVs: " + ", ".join(nouv))
        self.groups_hint.setText("\n".join(hints))
        self.groups_hint.setVisible(bool(hints))

    def _load_scene_points(self, path):
        """Keep raw (untransformed) points per object for parity checks and auto-align."""
        try:
            d = np.load(path)
            meta = json.loads(bytes(d["meta"]).decode())
            self.scene_pts = {o["name"]: (o["role"], o["file"], d[o["key"] + "_P"],
                                          o["role"] == "low") for o in meta["objects"]}
        except Exception as e:
            self.scene_pts = {}
            self.log(f"Could not read display meshes: {e}")

    # ------------------------------------------------------------------ viewer sync
    def world_matrix(self, name):
        fe = next((f for f in self.project.files if f.path == self.obj_file.get(name)), None)
        return xform.combined(fe.transform if fe else None, self.project.object_transforms.get(name))

    def sync_viewer(self):
        v = self.viewer3d.view
        v.set_matrices({n: self.world_matrix(n) for n in self.obj_file})
        v.set_hidden(self.project.hidden)
        s = self.collect_settings()
        v.set_low_info({g.low: {"set": texture_set_name(s, g.name), "cage_offset": g.cage_offset,
                                "cage": g.cage} for g in self.project.groups if g.low})

    def sync_viewer_cages(self):
        self.sync_viewer()

    def refresh_groups_tree(self):
        t = self.groups_tree
        sel = self._tree_selection_key()
        t.blockSignals(True)
        t.clear()
        info = {o["name"]: o for o in self.objects}
        hidden = set(self.project.hidden)
        loaded = bool(self.obj_file)
        reselect = None
        for gi, g in enumerate(self.project.groups):
            top = QTreeWidgetItem([g.name, "", f"{g.cage_offset:.4g}",
                                   f"{g.max_ray:.4g}" if g.max_ray else "∞"])
            top.setFlags(top.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            top.setCheckState(0, Qt.Checked if g.enabled else Qt.Unchecked)
            top.setToolTip(0, "Checkbox: include this group in the bake")
            top.setData(0, Qt.UserRole, ("group", gi))
            members = [n for n in [g.low, *g.highs, g.cage] if n]
            n_hidden = sum(n in hidden for n in members)
            top.setIcon(1, eye_icon("off" if n_hidden == len(members) and members else
                                    "part" if n_hidden else "on"))
            ok = bool(g.low and g.highs)
            top.setForeground(0, QColor("#e6e6e6") if ok else QColor("#e08a5b"))
            if g.name in self.parity and g.name not in self.parity_ignored:
                top.setIcon(0, warn_icon())
                top.setToolTip(0, "High and low don't line up — see the suggestion below the list")
            t.addTopLevelItem(top)
            if sel == ("group", g.name):
                reselect = top

            def child(kind, name):
                o = info.get(name)
                extra = f"  ({o['tris']:,} tris)" if o else ""
                c = QTreeWidgetItem([f"{kind}:  {name}{extra}"])
                c.setData(0, Qt.UserRole, (kind, gi, name))
                c.setIcon(1, eye_icon("off" if name in hidden else "on"))
                c.setToolTip(1, "Show/hide in the 3D viewer (Ctrl-click: show only this)")
                color = {"Low": "#8fc1ff", "High": "#ffc26b", "Cage": "#c79bff"}[kind]
                if loaded and o is None:
                    color = "#e06c6c"
                    c.setToolTip(0, "Not found in the loaded files (renamed or deleted?)")
                elif name in hidden:
                    color = "#6b6f78"
                if not xform.is_identity(self.project.object_transforms.get(name)):
                    c.setText(0, c.text(0) + "  ⟲")
                    c.setToolTip(0, "Has its own transform")
                c.setForeground(0, QColor(color))
                top.addChild(c)
                return c
            for kind, name in [("Low", g.low)] + [("High", h) for h in g.highs] + \
                    ([("Cage", g.cage)] if g.cage else []):
                c = child(kind, name)
                if sel == ("object", name):
                    reselect = c
            top.setExpanded(True)
        if reselect is not None:
            t.setCurrentItem(reselect)
        t.blockSignals(False)

    def _tree_selection_key(self):
        it = self.groups_tree.currentItem()
        d = it.data(0, Qt.UserRole) if it else None
        if not d:
            return None
        if d[0] == "group":
            return ("group", self.project.groups[d[1]].name) if d[1] < len(self.project.groups) else None
        return ("object", d[2])

    def tree_clicked(self, item, col):
        if col != 1:
            return
        d = item.data(0, Qt.UserRole)
        if not d:
            return
        g = self.project.groups[d[1]]
        names = [n for n in [g.low, *g.highs, g.cage] if n] if d[0] == "group" else [d[2]]
        hidden = set(self.project.hidden)
        if QApplication.keyboardModifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            everything = set(self.obj_file) | {n for gg in self.project.groups
                                                for n in [gg.low, *gg.highs, gg.cage] if n}
            hidden = everything - set(names)              # solo
        elif all(n in hidden for n in names):
            hidden -= set(names)
        else:
            hidden |= set(names)
        self.project.hidden = sorted(hidden)
        self.refresh_groups_tree()
        self.viewer3d.view.set_hidden(self.project.hidden)

    def _current_group(self):
        it = self.groups_tree.currentItem()
        if it is None:
            return None
        d = it.data(0, Qt.UserRole)
        return self.project.groups[d[1]] if d else None

    def group_selected(self, cur, _prev):
        g = self._current_group()
        if g:
            self._loading = True
            self.g_offset.setValue(g.cage_offset)
            self.g_maxray.setValue(g.max_ray)
            self._loading = False
        d = cur.data(0, Qt.UserRole) if cur else None
        if d and d[0] != "group":
            self.files_table.blockSignals(True)
            self.files_table.clearSelection()
            self.files_table.blockSignals(False)
            self.set_target(("object", d[2]))
        elif d:
            self.set_target(None)

    def file_selected(self):
        rows = sorted({i.row() for i in self.files_table.selectedIndexes()})
        if rows and rows[0] < len(self.project.files):
            self.groups_tree.blockSignals(True)
            self.groups_tree.setCurrentItem(None)
            self.groups_tree.blockSignals(False)
            self.set_target(("file", self.project.files[rows[0]].path))

    # ------------------------------------------------------------------ transforms
    def _target_group(self):
        """The group the current selection belongs to (for auto-align)."""
        g = self._current_group()
        if g is not None:
            return g
        if self.target and self.target[0] == "object":
            for g in self.project.groups:
                if self.target[1] in [g.low, *g.highs, g.cage]:
                    return g
        return None

    def set_target(self, target):
        self.target = target
        grp = self._target_group()
        can_align = bool(grp and grp.low and grp.highs and self.scene_pts)
        if target is None:
            self.xf.set_target(None, can_align=can_align)
        elif target[0] == "file":
            fe = self._file_entry(target[1])
            n = sum(1 for f in self.obj_file.values() if f == target[1])
            self.xf.set_target(f"File: {os.path.basename(target[1])}  — moves all {n} mesh(es) "
                               f"in it", fe.transform if fe else None, can_align)
        else:
            name = target[1]
            src = os.path.basename(self.obj_file.get(name, "")) or "?"
            self.xf.set_target(f"Object: {name}  — applied before its file's transform ({src})",
                               self.project.object_transforms.get(name), can_align)

    def _file_entry(self, path):
        return next((f for f in self.project.files if f.path == path), None)

    def _transform_state(self):
        return (copy.deepcopy([f.transform for f in self.project.files]),
                copy.deepcopy(self.project.object_transforms))

    def _restore_transform_state(self, st):
        files, objs = st
        for fe, tr in zip(self.project.files, files):
            fe.transform = copy.deepcopy(tr)
        self.project.object_transforms = copy.deepcopy(objs)

    def _push_undo(self, coalesce=False):
        """Remember the state before an edit (rapid spinbox edits count as one step)."""
        now = time.monotonic()
        if not (coalesce and now - self._last_edit < 1.0 and self.undo_stack):
            self.undo_stack.append(self._xf_state)
            del self.undo_stack[:-100]
            self.redo_stack.clear()
        self._last_edit = now if coalesce else 0.0

    def _transforms_changed(self):
        self._xf_state = self._transform_state()
        self.xf.set_undo_state(bool(self.undo_stack), bool(self.redo_stack))
        self.sync_viewer()
        self.refresh_groups_tree()
        self.set_target(self.target)
        self.schedule_parity()

    def _write_target(self, tr):
        if not self.target:
            return
        tr = xform.normalized(tr)
        if self.target[0] == "file":
            fe = self._file_entry(self.target[1])
            if fe:
                fe.transform = tr
        else:
            if xform.is_identity(tr):
                self.project.object_transforms.pop(self.target[1], None)
            else:
                self.project.object_transforms[self.target[1]] = tr

    def on_transform_edited(self, tr):
        if not self.target:
            return
        self._push_undo(coalesce=True)
        self._write_target(tr)
        self._xf_state = self._transform_state()
        self.xf.set_undo_state(bool(self.undo_stack), bool(self.redo_stack))
        self.sync_viewer()                         # live, cheap
        self._xf_settle.start()                    # outliner / parity after the edit settles

    def reset_transform(self):
        if self.target:
            self._push_undo()
            self._write_target(xform.identity())
            self._transforms_changed()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self._transform_state())
            self._restore_transform_state(self.undo_stack.pop())
            self._last_edit = 0.0
            self._transforms_changed()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self._transform_state())
            self._restore_transform_state(self.redo_stack.pop())
            self._last_edit = 0.0
            self._transforms_changed()

    def set_group_value(self, attr, v):
        if self._loading:
            return
        if attr == "cage_offset":
            self.project.default_cage_offset = v
        else:
            self.project.default_max_ray = v
        g = self._current_group()
        if g:
            setattr(g, attr, v)
            self._update_group_row(g)
            if attr == "cage_offset":
                self.sync_viewer()

    def _update_group_row(self, g):
        gi = self.project.groups.index(g)
        top = self.groups_tree.topLevelItem(gi)
        if top:
            self.groups_tree.blockSignals(True)
            top.setText(1, f"{g.cage_offset:.4g}")
            top.setText(2, f"{g.max_ray:.4g}" if g.max_ray else "∞")
            self.groups_tree.blockSignals(False)

    def apply_offsets_all(self):
        for g in self.project.groups:
            g.cage_offset = self.g_offset.value()
            g.max_ray = self.g_maxray.value()
        self.refresh_groups_tree()
        self.sync_viewer_cages()

    def group_item_changed(self, item, col):
        d = item.data(0, Qt.UserRole)
        if not d or d[0] != "group":
            return
        g = self.project.groups[d[1]]
        g.enabled = item.checkState(0) == Qt.Checked
        new = item.text(0).strip()
        if new and new != g.name:
            g.name = new

    def groups_menu(self, pos):
        it = self.groups_tree.itemAt(pos)
        if not it:
            return
        d = it.data(0, Qt.UserRole)
        m = QMenu(self)
        if d[0] in ("High", "Cage"):
            kind, gi, name = d
            src = self.project.groups[gi]
            if kind == "High":
                mv = m.addMenu("Move to group")
                for gj, g in enumerate(self.project.groups):
                    if gj != gi:
                        mv.addAction(g.name, lambda gj=gj: self._move_high(gi, gj, name))
                m.addAction("Remove from group", lambda: (src.highs.remove(name), self.refresh_groups_tree()))
            else:
                m.addAction("Remove cage", lambda: (setattr(src, "cage", ""), self.refresh_groups_tree()))
        if d[0] in ("Low", "High", "Cage") and d[2] in self.project.object_transforms:
            m.addAction("Clear this object's transform", lambda n=d[2]: self._clear_obj_transform(n))
        if d[0] == "group" or d[0] in ("Low", "High", "Cage"):
            g = self.project.groups[d[1]]
            m.addAction("Auto-align highs to low", lambda g=g: self.auto_align_group(g))
        if d[0] == "group":
            gi = d[1]
            m.addAction("Bake only this group", lambda: self.bake(only=gi))
            m.addAction("Rename", lambda: self.groups_tree.editItem(it, 0))
            m.addAction("Delete group", lambda: (self.project.groups.pop(gi), self.refresh_groups_tree()))
        m.exec(self.groups_tree.viewport().mapToGlobal(pos))

    def _clear_obj_transform(self, name):
        self._push_undo()
        self.project.object_transforms.pop(name, None)
        self._transforms_changed()

    def _move_high(self, gi, gj, name):
        self.project.groups[gi].highs.remove(name)
        self.project.groups[gj].highs.append(name)
        self.refresh_groups_tree()

    # ------------------------------------------------------------------ parity / auto-align
    def _group_points(self, g):
        """World-space points of a group's low (triangles) and highs, or None if not loaded."""
        if g.low not in self.scene_pts or not g.highs:
            return None
        highs = [h for h in g.highs if h in self.scene_pts]
        if not highs:
            return None
        low = xform.apply_points(self.world_matrix(g.low), self.scene_pts[g.low][2])
        high = np.concatenate([xform.apply_points(self.world_matrix(h), self.scene_pts[h][2])
                               for h in highs])
        return low, high

    def schedule_parity(self):
        """Check every group in the background; results arrive in _parity_results."""
        self._parity_gen += 1
        gen = self._parity_gen
        jobs = []
        for g in self.project.groups:
            pts = self._group_points(g)
            if pts is not None:
                jobs.append((g.name, pts))
        if not jobs:
            self.parity = {}
            self._show_parity()
            return

        def run():
            res = {}
            for name, (lo, hi) in jobs:
                try:
                    r = align.check(lo, hi)
                except Exception as e:      # never let a check break the app
                    r = None
                    print("parity check failed", name, e)
                if r is not None:
                    res[name] = r
            self.parity_done.fire.emit((gen, res))
        threading.Thread(target=run, daemon=True).start()

    def _parity_results(self, payload):
        gen, res = payload
        if gen != self._parity_gen:          # transforms changed meanwhile: a newer check runs
            return
        self.parity = res
        self._show_parity()
        self.refresh_groups_tree()

    def _show_parity(self):
        while self.parity_layout.count():
            w = self.parity_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        shown = [(n, r) for n, r in self.parity.items() if n not in self.parity_ignored]
        for name, r in shown:
            fix_scope, _f, fix_D = self._fix_scope(name, r)
            row = QFrame()
            row.setObjectName("parity")
            h = QHBoxLayout(row)
            h.setContentsMargins(8, 5, 6, 5)
            what = {"scale": "different sizes", "axes": "rotated", "offset": "offset from each other",
                    "orientation": "rotated"}.get(r["reason"], "misaligned")
            txt = (f"<b>{_html(name)}</b>: high and low look {what}. "
                   f"Suggested fix: {_html(align.describe(fix_D, r['snapped']))}"
                   f"{' — applies to the whole file' if fix_scope == 'file' else ''}.")
            if r["ambiguous"]:
                txt += " <i>The model is symmetric, check the result.</i>"
            lab = QLabel(txt)
            lab.setWordWrap(True)
            h.addWidget(lab, 1)
            fix = QPushButton("Fix")
            fix.clicked.connect(lambda _=False, n=name: self.apply_parity_fix(n))
            ign = QToolButton(text="✕")
            ign.setToolTip("Ignore this warning")
            ign.clicked.connect(lambda _=False, n=name: (self.parity_ignored.add(n), self._show_parity(),
                                                        self.refresh_groups_tree()))
            h.addWidget(fix)
            h.addWidget(ign)
            self.parity_layout.addWidget(row)
        self.parity_box.setVisible(bool(shown))

    def _fix_scope(self, group_name, r):
        """Where a fix should go. Returns (scope, file, D):
        ("file", path, D) when the high file is clearly the cause: every group whose highs come
        from that file is off and one correction fits all of them (symmetric models can make the
        per-group suggestions differ, so each candidate is tried on every group);
        ("objects", None, r["D"]) otherwise: just this group's high polys."""
        g = next((x for x in self.project.groups if x.name == group_name), None)
        if g is None:
            return "objects", None, r["D"]
        files = {self.obj_file.get(h) for h in g.highs}
        if len(files) != 1 or None in files:
            return "objects", None, r["D"]
        f = files.pop()
        in_file = [n for n, p in self.obj_file.items() if p == f]
        if any(n in self.project.object_transforms for n in in_file) or \
                any(self.obj_file.get(x.low) == f for x in self.project.groups):
            return "objects", None, r["D"]           # overrides / lows in that file: be local
        sharing = [x for x in self.project.groups if any(self.obj_file.get(h) == f for h in x.highs)]
        if any(x.name not in self.parity for x in sharing):
            return "objects", None, r["D"]           # some group in that file is already fine
        cands = [r["D"]] + [self.parity[x.name]["D"] for x in sharing if x.name != group_name]
        rng = np.random.default_rng(0)
        for D in cands:
            ok = True
            for x in sharing:
                pts = self._group_points(x)
                if pts is None:
                    continue
                lo, hi = pts
                L = align.sample_triangles(lo, 600, rng)
                H = align.subsample(xform.apply_points(D, hi), 2500, rng)
                diag = float(np.linalg.norm(lo.max(0) - lo.min(0)))
                if align.fit_error(L, H) / diag > self.parity[x.name]["error_after"] * 1.25 + 0.005:
                    ok = False
                    break
            if ok:
                return "file", f, D
        return "objects", None, r["D"]

    def _apply_correction(self, g, D, scope=None):
        """Apply world-space correction D to a file (scope "file") or to the group's highs."""
        scope, f = scope if scope else ("objects", None)
        if scope == "file":
            fe = self._file_entry(f)
            fe.transform = xform.decompose(D @ xform.matrix(fe.transform))
        else:
            for h in g.highs:
                # world = F @ O; want D @ F @ O = F @ O'  ->  O' = F^-1 @ D @ F @ O
                F = xform.matrix(self._file_entry(self.obj_file.get(h)).transform
                                 if self._file_entry(self.obj_file.get(h)) else None)
                cur = xform.matrix(self.project.object_transforms.get(h))
                tr = xform.decompose(np.linalg.inv(F) @ D @ F @ cur)
                if xform.is_identity(tr):
                    self.project.object_transforms.pop(h, None)
                else:
                    self.project.object_transforms[h] = tr

    def apply_parity_fix(self, name):
        r = self.parity.get(name)
        g = next((x for x in self.project.groups if x.name == name), None)
        if not r or not g:
            return
        scope, f, D = self._fix_scope(name, r)
        self._push_undo()
        if scope == "file":
            self._apply_correction(g, D, (scope, f))
            for x in self.project.groups:             # the same file fix covers these too
                if x.name in self.parity and any(self.obj_file.get(h) == f for h in x.highs):
                    self.parity.pop(x.name, None)
            self.log(f"Fixed {os.path.basename(f)}: {align.describe(D, r['snapped'])}")
        else:
            self._apply_correction(g, r["D"])
            self.parity.pop(name, None)
            self.log(f"Fixed {name}: {align.describe(r['D'], r['snapped'])}")
        self._show_parity()
        self._transforms_changed()
        self.statusBar().showMessage("Alignment fixed — Ctrl+Z to undo", 6000)

    def auto_align_group(self, g):
        if g is None:
            QMessageBox.information(self, APP_NAME, "Select a group (or one of its meshes) first.")
            return
        pts = self._group_points(g)
        if pts is None:
            QMessageBox.information(self, APP_NAME, "This group needs a loaded low and high poly.")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            r = align.auto_align(*pts)
        finally:
            QApplication.restoreOverrideCursor()
        if r["error_after"] >= r["error_before"] * 0.98:
            self.statusBar().showMessage(f"{g.name}: already aligned, nothing to change", 6000)
            return
        self._push_undo()
        self._apply_correction(g, r["D"])
        self.parity.pop(g.name, None)
        self._show_parity()
        self._transforms_changed()
        msg = f"{g.name}: {align.describe(r['D'], r['snapped'])}"
        self.log("Auto-align " + msg)
        if not r["confident"]:
            QMessageBox.warning(self, APP_NAME, msg + "\n\nThe fit is not very close, please check "
                                "the result in the 3D viewer (Ctrl+Z to undo).")
        elif r["ambiguous"]:
            self.statusBar().showMessage(msg + "  — symmetric model, check the result (Ctrl+Z to undo)",
                                         10000)
        else:
            self.statusBar().showMessage(msg + "  (Ctrl+Z to undo)", 8000)

    # ------------------------------------------------------------------ export corrected meshes
    def export_dialog(self):
        files = [fe.path for fe in self.project.files
                 if not xform.is_identity(fe.transform)
                 or any(f == fe.path and not xform.is_identity(self.project.object_transforms.get(n))
                        for n, f in self.obj_file.items())]
        if not files:
            QMessageBox.information(self, APP_NAME, "No file has a transform yet, so there is "
                                    "nothing to export. Fix the alignment first.")
            return
        dlg = ExportDialog(self, files)
        if dlg.exec() != ExportDialog.Accepted:
            return
        self._export_switch = dlg.switch.isChecked()
        self._export_only = dlg.selected()
        if not self._export_only:
            return
        self.collect_settings()
        self.set_busy(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Exporting corrected meshes…")
        extra = ["--format", dlg.fmt.currentData(), "--suffix", dlg.suffix.text() or "_aligned"]
        for p in self._export_only:
            extra += ["--only", p]
        self.worker.start("export", self.project, extra)

    def _exported(self, files):
        if not files:
            return
        lines = "\n".join(f"  {os.path.basename(x['source'])}  →  {x['path']}" for x in files)
        self.log("Exported corrected meshes:\n" + lines)
        if getattr(self, "_export_switch", False):
            self._push_undo()
            for x in files:
                fe = self._file_entry(x["source"])
                if fe is None:
                    continue
                for n, f in list(self.obj_file.items()):
                    if f == fe.path:
                        self.project.object_transforms.pop(n, None)
                fe.path = x["path"]
                fe.transform = xform.identity()
            self.undo_stack.clear()                  # file paths changed: old states don't apply
            self.redo_stack.clear()
            self.refresh_files_table()
            self._restore_groups = list(self.project.groups)
            QTimer.singleShot(0, lambda: self.reload_meshes(quiet=True))
        QMessageBox.information(self, APP_NAME, f"Exported {len(files)} corrected file(s):\n\n" + lines)

    # ------------------------------------------------------------------ auto reload
    def _auto_reload_toggled(self, on):
        self.qs.setValue("auto_reload", bool(on))
        self.watcher.enabled = bool(on)

    def on_files_changed(self, paths):
        if not self.auto_reload.isChecked():
            return
        self._reload_queue.update(paths)
        if not self.worker.busy():
            self.reload_meshes()

    def reload_meshes(self, quiet=False):
        """Re-import the mesh files, keeping groups, transforms, visibility and the camera."""
        if not self.project.files:
            return
        if self.worker.busy():
            self._reload_queue.update(f.path for f in self.project.files)
            return
        changed = sorted(self._reload_queue) or [f.path for f in self.project.files]
        self._reload_queue.clear()
        self._reloading = {"files": changed, "quiet": quiet}
        if getattr(self, "_restore_groups", None) is None:
            self._restore_groups = list(self.project.groups)
        self._old_groups = {g.name: g for g in self.project.groups}
        self.collect_settings()
        self.progress.setRange(0, 0)
        self.progress.setFormat("Reloading meshes…")
        self.set_busy(True)
        self.worker.start("scan", self.project, ["--mesh-out", self.mesh_path])

    def _after_reload(self):
        info, self._reloading = self._reloading, None
        self.watcher.mark_loaded()
        if info and not info["quiet"]:
            names = ", ".join(os.path.basename(p) for p in info["files"])
            self.log(f"Reloaded {names}")
            if self.maps_present:
                self.stale_label.setText(f"⚠ {names} reloaded — maps may be out of date")
                self.stale_label.setToolTip("The baked maps were made from the previous version of "
                                            "the mesh. Bake again to update them.")
                self.stale_label.setVisible(True)
                self.statusBar().showMessage(f"{names} reloaded — maps may be out of date")
            else:
                self.statusBar().showMessage(f"{names} reloaded")

    # ------------------------------------------------------------------ bake / render
    def set_busy(self, busy):
        self.bake_btn.setEnabled(not busy)
        self.render_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)

    def bake(self, only=None):
        s = self.collect_settings()
        problems = []
        if not self.project.files:
            problems.append("Add some mesh files first (drag & drop them onto the window).")
        if not any(g.enabled and g.highs for g in self.project.groups):
            problems.append("No bake group has both a low and a high poly.")
        if not enabled_maps(s):
            problems.append("Pick at least one map.")
        if not s.output_dir:
            problems.append("Choose an output folder.")
        if problems:
            QMessageBox.warning(self, APP_NAME, "\n".join(problems))
            return
        proj = self.project
        if only is not None:
            proj = Project.from_dict(self.project.to_dict())
            for i, g in enumerate(proj.groups):
                g.enabled = i == only
        self.thumbs.clear()
        self.map_previews.clear()
        for f in os.listdir(self.preview_dir):
            if f.endswith(".png") and f != "render.png":
                os.remove(os.path.join(self.preview_dir, f))
        self.progress.setRange(0, 0)
        self.progress.setFormat("Preparing…")
        self.set_busy(True)
        self.tabs.setCurrentIndex(0)
        self.log(f"--- Bake started: {', '.join(MAPS[k][0] for k in enabled_maps(s))}")
        self.worker.start("bake", proj, ["--preview-dir", self.preview_dir])

    def render_preview(self):
        if self.worker.busy() or not self.project.groups:
            return
        self.collect_settings()
        self.progress.setRange(0, 0)
        self.progress.setFormat("Rendering preview…")
        self.set_busy(True)
        self.tabs.setCurrentIndex(2)
        self.worker.start("render", self.project, ["--out", self.render_path,
                                                   "--yaw", str(self.yaw.value()),
                                                   "--pitch", str(self.pitch.value()),
                                                   "--samples", str(self.r_samples.value()),
                                                   "--show", self.r_show.currentData()])

    def cancel(self):
        self.worker.cancel()
        self.log("Cancelled.")

    def on_event(self, ev):
        e = ev.get("event")
        if e == "progress":
            self.progress.setRange(0, ev["total"])
            self.progress.setValue(ev["step"])
            self.progress.setFormat(f"{ev['map']} — {ev['group']}  (%v / %m)")
        elif e == "log":
            self.log(ev["message"])
        elif e == "map_done":
            self.log(f"  wrote {ev['path']}", dim=True)
            if ev.get("preview"):
                self.viewer3d.view.set_texture(ev["set"], ev["map"], ev["preview"])
                label = f"{ev['set']}  {MAPS[ev['map']][0]}"
                it = QListWidgetItem(QIcon(ev["preview"]), label)
                it.setData(Qt.UserRole, ev["preview"])
                it.setData(Qt.UserRole + 1, ev["path"])
                it.setToolTip(ev["path"])
                self.thumbs.addItem(it)
                if self.thumbs.count() == 1:
                    self.thumbs.setCurrentItem(it)
        elif e == "scan":
            self.apply_scan(ev)
        elif e == "exported":
            self._exported(ev.get("files", []))
        elif e == "render":
            self.render_view.set_image(ev["path"])
        elif e == "done":
            self.maps_present = True
            self.stale_label.setVisible(False)
            self.log(f"--- Done: {len(ev['files'])} file(s) in {ev['seconds']}s")
            self.statusBar().showMessage(f"Baked {len(ev['files'])} maps in {ev['seconds']}s → "
                                         f"{self.project.settings.output_dir}")
            self._pending_render = self.auto_render.isChecked()
        elif e == "error":
            self.log("ERROR: " + ev["message"])
            self.log(ev.get("trace", ""), dim=True)
            QMessageBox.critical(self, APP_NAME, ev["message"])

    def on_worker_finished(self, cmd, code):
        self.set_busy(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if code == 0 else 0)
        self.progress.setFormat({"bake": "Bake finished", "scan": "Meshes loaded",
                                 "render": "Preview rendered",
                                 "export": "Export finished"}.get(cmd, "Done") if code == 0
                                else "Stopped")
        if cmd == "scan":
            self.refresh_groups_tree()
            if self._reloading:
                self._after_reload()
            else:
                self.watcher.mark_loaded()
        if self._reload_queue and not self.worker.busy():
            QTimer.singleShot(0, self.reload_meshes)
        if cmd == "scan":
            selftest = os.environ.get("HLBAKE_SELFTEST")
            if selftest:   # smoke test used by CI: report groups, screenshot, quit
                def finish():
                    self.grab().save(os.path.splitext(selftest)[0] + ".png")
                    with open(selftest, "w") as fh:
                        base = getattr(sys, "_MEIPASS", "")
                        sw = [d for d in (base, os.path.join(base, "PySide6"))
                              if base and os.path.exists(os.path.join(d, "opengl32sw.dll"))]
                        json.dump({"exit": code, "viewer_gl": self.viewer3d.view.ok,
                                   "gl_renderer": self.viewer3d.view.renderer,
                                   "gl_error": self.viewer3d.view.error,
                                   "software_gl": os.environ.get("HLBAKE_SOFTWARE_GL") == "1",
                                   "gl_probe": GL_PROBE, "opengl32sw_in": sw,
                                   "gl_diag": __import__("hlbake.viewport", fromlist=["GL_DIAG"]).GL_DIAG,
                                   "groups": [g.__dict__ for g in self.project.groups]}, fh)
                    QApplication.instance().quit()
                QTimer.singleShot(2500, finish)
        if self._pending_render:
            self._pending_render = False
            QTimer.singleShot(50, self.render_preview)

    def log(self, text, dim=False):
        if not text:
            return
        if dim:
            self.logbox.appendHtml(f'<span style="color:#7d8590">{_html(text)}</span>')
        else:
            self.logbox.appendPlainText(text)

    # ------------------------------------------------------------------ project files
    def update_title(self):
        name = os.path.basename(self.project_path) if self.project_path else "Untitled"
        self.setWindowTitle(f"{name} — {APP_NAME} {APP_VERSION}")

    def new_project(self):
        self.project = Project()
        self.project_path = None
        self.objects = []
        self.obj_file, self.scene_pts, self.parity = {}, {}, {}
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.maps_present = False
        self.stale_label.setVisible(False)
        self.watcher.watch([])
        self._show_parity()
        self.set_target(None)
        self.thumbs.clear()
        self.load_project_into_ui()

    def open_project(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open project", self.qs.value("last_dir", ""),
                                                  f"HL Baker project (*{PROJECT_EXT})")
        if not path:
            return
        try:
            self.project = Project.load(path)
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Could not open project:\n{e}")
            return
        self.project_path = path
        saved_groups = list(self.project.groups)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._xf_state = self._transform_state()
        self.parity, self.parity_ignored = {}, set()
        self.load_project_into_ui()
        if self.project.files:
            # re-import to get object info, but keep the saved groups
            self._old_groups = {g.name: g for g in saved_groups}
            self.collect_settings()
            self.set_busy(True)
            self.progress.setRange(0, 0)
            self.progress.setFormat("Loading meshes…")
            self._restore_groups = saved_groups
            self.viewer3d.view.clear_textures()
            self.worker.start("scan", self.project, ["--mesh-out", self.mesh_path])
        self.load_existing_maps()

    def load_existing_maps(self):
        """Show already-baked maps from the output folder (formats Qt can read)."""
        s = self.project.settings
        ext = FORMATS[s.file_format][2]
        if not s.output_dir or not os.path.isdir(s.output_dir) or ext == "exr":
            return
        self.thumbs.clear()
        suffix_to_key = {v[1]: k for k, v in MAPS.items()}
        for f in sorted(os.listdir(s.output_dir)):
            if f.endswith("." + ext):
                p = os.path.join(s.output_dir, f)
                stem = os.path.splitext(f)[0]
                for suf in sorted(suffix_to_key, key=len, reverse=True):
                    if stem.endswith("_" + suf):
                        self.viewer3d.view.set_texture(stem[:-len(suf) - 1], suffix_to_key[suf], p)
                        break
                it = QListWidgetItem(QIcon(p), os.path.splitext(f)[0])
                it.setData(Qt.UserRole, p)
                it.setData(Qt.UserRole + 1, p)
                self.thumbs.addItem(it)
                self.maps_present = True

    def save_project(self):
        if not self.project_path:
            return self.save_project_as()
        self.collect_settings()
        self.project.save(self.project_path)
        self.statusBar().showMessage(f"Saved {self.project_path}", 4000)

    def save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save project", self.qs.value("last_dir", ""),
                                              f"HL Baker project (*{PROJECT_EXT})")
        if path:
            if not path.endswith(PROJECT_EXT):
                path += PROJECT_EXT
            self.project_path = path
            self.save_project()
            self.update_title()

    def pick_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.w["output_dir"].text())
        if d:
            self.w["output_dir"].setText(d)

    def show_naming_help(self):
        QMessageBox.information(self, "Naming conventions", (
            "Groups are built from mesh names, like Toolbag's Quick Loader:\n\n"
            "  crate_low     crate_high     crate_high1     crate_cage\n\n"
            "Suffixes _low/_lp/_lo, _high/_hp/_hi and _cage are recognised (any case, with "
            "_ . - or space as separator, optionally followed by a number).\n\n"
            "If you load files with the role Low or High, every mesh in that file gets that role; "
            "highs whose names don't match a low are assigned to the low they overlap most.\n\n"
            "Right-click a high poly in the group list to move it to another group."))

    def show_about(self):
        QMessageBox.about(self, f"About {APP_NAME}", (
            f"<b>{APP_NAME} {APP_VERSION}</b><br>High-to-low poly texture baker.<br><br>"
            "Baking is powered by Blender's Cycles renderer (bpy module).<br>"
            "Blender is © the Blender Foundation and licensed under the GNU GPL; "
            "this program is distributed under the GNU GPL v3 or later."))

    def closeEvent(self, e):
        self.worker.cancel()
        super().closeEvent(e)


def _html(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class _Relay(QObject):
    """Carries results from a background thread to the GUI thread."""
    fire = Signal(object)


_ICONS = {}


def eye_icon(state):
    """Eye icons drawn in code: on / off (closed) / part (some members hidden)."""
    if state in _ICONS:
        return _ICONS[state]
    from PySide6.QtGui import QPen, QPainterPath
    pm = QPixmap(32, 32)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    col = QColor("#d8dbe0") if state == "on" else QColor("#8a8f98") if state == "part" else QColor("#5d6168")
    pen = QPen(col, 2.6)
    p.setPen(pen)
    path = QPainterPath()
    path.moveTo(3, 16)
    path.quadTo(16, 3 if state != "off" else 26, 29, 16)
    if state != "off":
        path.quadTo(16, 29, 3, 16)
    p.drawPath(path)
    if state != "off":
        p.setBrush(col)
        p.drawEllipse(12, 12, 8, 8)
    else:
        for x in (8, 16, 24):
            p.drawLine(x, 21 + (2 if x == 16 else 0), x - (2 if x < 16 else -2 if x > 16 else 0), 26)
    p.end()
    _ICONS[state] = QIcon(pm)
    return _ICONS[state]


def warn_icon():
    if "warn" not in _ICONS:
        pm = QPixmap(32, 32)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor("#e0a03c"))
        p.setPen(Qt.NoPen)
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QPolygonF
        p.drawPolygon(QPolygonF([QPointF(16, 3), QPointF(30, 28), QPointF(2, 28)]))
        p.setPen(QColor("#1d1f22"))
        f = p.font()
        f.setBold(True)
        f.setPixelSize(18)
        p.setFont(f)
        p.drawText(pm.rect().adjusted(0, 7, 0, 0), Qt.AlignCenter, "!")
        p.end()
        _ICONS["warn"] = QIcon(pm)
    return _ICONS["warn"]


class ExportDialog(QDialog):
    def __init__(self, parent, files):
        super().__init__(parent)
        self.setWindowTitle("Export corrected meshes")
        v = QVBoxLayout(self)
        v.addWidget(QLabel("Write copies of these files with the alignment applied to the geometry.\n"
                           "Object names, pivots and materials are kept; the originals are untouched."))
        self.checks = []
        for path in files:
            c = QCheckBox(os.path.basename(path))
            c.setToolTip(path)
            c.setChecked(True)
            c.path = path
            self.checks.append(c)
            v.addWidget(c)
        form = QFormLayout()
        self.fmt = combo([("SAME", "Same as the original"), ("FBX", "FBX"), ("OBJ", "OBJ"),
                          ("GLB", "glTF binary (.glb)"), ("GLTF", "glTF (.gltf + .bin)")])
        form.addRow("Format", self.fmt)
        self.suffix = QLineEdit("_aligned")
        form.addRow("File name suffix", self.suffix)
        v.addLayout(form)
        self.switch = QCheckBox("Use the exported files in this project (their transforms reset)")
        self.switch.setChecked(True)
        v.addWidget(self.switch)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Export")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def selected(self):
        return [c.path for c in self.checks if c.isChecked()]


# ---------------------------------------------------------------------------
STYLE = """
QWidget { font-size: 10pt; }
QGroupBox#section { font-weight: 600; font-size: 10.5pt; border: 1px solid #3a3d44; border-radius: 6px;
    margin-top: 14px; padding: 10px 8px 8px 8px; background: #26282c; }
QGroupBox#section::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #f0a94b; }
QGroupBox#inner { font-weight: normal; border: 1px solid #34373d; border-radius: 4px; margin-top: 10px;
    padding: 8px 6px 6px 6px; }
QGroupBox#inner::title { subcontrol-origin: margin; left: 8px; color: #9aa1ab; }
QLabel#sub { color: #9aa1ab; margin-top: 4px; }
QLabel#hint { color: #c9a26b; font-size: 9pt; }
QPushButton { padding: 5px 10px; border-radius: 4px; background: #34373d; border: 1px solid #44474f; }
QPushButton:hover { background: #3d4048; }
QPushButton:disabled { color: #6b6f78; }
QPushButton#bake { background: #d9822b; color: #111; font-weight: 700; font-size: 12pt; border: none; }
QPushButton#bake:hover { background: #ec9640; }
QPushButton#bake:disabled { background: #5a4630; color: #8a7a66; }
QProgressBar { border: 1px solid #3a3d44; border-radius: 4px; text-align: center; height: 18px;
    background: #1f2124; }
QProgressBar::chunk { background: #d9822b; border-radius: 3px; }
QTreeWidget, QTableWidget, QListWidget, QPlainTextEdit { background: #1f2124; border: 1px solid #34373d; }
QTabWidget::pane { border: 1px solid #34373d; }
QTabBar::tab { padding: 6px 14px; }
QFrame#parity { background: #3a2f1f; border: 1px solid #6b5228; border-radius: 4px; }
QLabel#stale { color: #f0b45a; padding-right: 8px; }
"""


def dark_palette(app):
    app.setStyle("Fusion")
    p = QPalette()
    base, alt, text = QColor(31, 33, 36), QColor(38, 40, 44), QColor(225, 227, 230)
    p.setColor(QPalette.Window, QColor(33, 35, 39))
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, base)
    p.setColor(QPalette.AlternateBase, alt)
    p.setColor(QPalette.ToolTipBase, alt)
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, QColor(52, 55, 61))
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.Highlight, QColor(217, 130, 43))
    p.setColor(QPalette.HighlightedText, QColor(20, 20, 20))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(110, 114, 120))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(110, 114, 120))
    app.setPalette(p)
    app.setStyleSheet(STYLE)


GL_PROBE = {}


def _gl33_available():
    from PySide6.QtGui import QOpenGLContext, QSurfaceFormat
    ctx = QOpenGLContext()
    ctx.setFormat(QSurfaceFormat.defaultFormat())
    created = ctx.create()
    f = ctx.format()
    GL_PROBE.update(created=created, version=f"{f.majorVersion()}.{f.minorVersion()}")
    if not created:
        return False
    return (f.majorVersion(), f.minorVersion()) >= (3, 3)


def _relaunch_with_software_gl(argv):
    """Windows: start again with Qt's bundled software OpenGL (opengl32sw.dll)."""
    import subprocess
    # Mesa: use the llvmpipe CPU renderer directly (skip its Direct3D 12 path, which needs dxil.dll)
    env = dict(os.environ, HLBAKE_SOFTWARE_GL="1", GALLIUM_DRIVER="llvmpipe")
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, *argv]
    else:
        main_py = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "hlbake_main.py")
        cmd = [sys.executable, main_py, *argv]
    subprocess.Popen(cmd, env=env)


def _selftest_log():
    """When self-testing, log progress and native crashes to <selftest>.log (windowed apps have
    no console)."""
    st = os.environ.get("HLBAKE_SELFTEST")
    if not st:
        return lambda *a: None
    import faulthandler
    path = os.path.splitext(st)[0] + ".log"
    fh = open(path, "a", buffering=1)
    faulthandler.enable(fh)
    if sys.stderr is None:
        sys.stderr = fh
    if sys.stdout is None:
        sys.stdout = fh
    def log(*a):
        fh.write(f"[pid {os.getpid()}] " + " ".join(str(x) for x in a) + "\n")
    return log


def main(argv=None):
    from PySide6.QtCore import QCoreApplication
    slog = _selftest_log()
    slog("start", "software_gl=" + os.environ.get("HLBAKE_SOFTWARE_GL", "0"))
    # Only read by Mesa (Qt's Windows fallback renderer, opengl32sw.dll): use its stable CPU
    # renderer rather than the D3D12 translation layer. Real GPU drivers ignore it.
    if sys.platform == "win32":
        os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
    request_gl_format()
    software_gl = os.environ.get("HLBAKE_SOFTWARE_GL") == "1"
    if software_gl:
        QCoreApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
    app = QApplication(sys.argv[:1])
    slog("QApplication created")
    if sys.platform == "win32" and not software_gl and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        try:
            ok = _gl33_available()
            slog("GL 3.3 probe:", ok, GL_PROBE)
            if not ok:
                _relaunch_with_software_gl(list(argv or []))
                slog("relaunched with software OpenGL")
                return 0
        except Exception as e:
            slog("probe failed:", e)
    app.setApplicationName(APP_NAME)
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.exists(os.path.join(base, "icon.png")):
        app.setWindowIcon(QIcon(os.path.join(base, "icon.png")))
    dark_palette(app)
    win = MainWindow()
    slog("window created")
    win.show()
    slog("window shown")
    args = [a for a in (argv or []) if not a.startswith("-")]
    projects = [a for a in args if a.endswith(PROJECT_EXT)]
    if projects:
        win.open_project(projects[0])
    elif args:
        win.add_files(args)
    return app.exec()
