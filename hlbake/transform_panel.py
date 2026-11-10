# SPDX-License-Identifier: GPL-3.0-or-later
"""Numeric transform editor for a file or an object (location / rotation / scale + quick fixes)."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QLabel,
                               QDoubleSpinBox, QPushButton, QToolButton, QCheckBox, QMenu)

from . import xform

QUICK_SCALES = [(100.0, "×100", "m → cm"), (0.01, "÷100", "cm → m"), (2.54, "×2.54", "in → cm"),
                (1 / 2.54, "÷2.54", "cm → in"), (10.0, "×10", ""), (0.1, "÷10", "")]


def _spin(lo, hi, dec, step, suffix=""):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setSuffix(suffix)
    w.setKeyboardTracking(False)
    w.setMinimumWidth(70)
    return w


class TransformPanel(QWidget):
    """Emits changed(transform dict) whenever the user edits the fields or uses a quick button.
    The owner decides what the transform belongs to (set_target)."""
    changed = Signal(dict)
    reset_requested = Signal()
    align_requested = Signal()
    export_requested = Signal()
    undo_requested = Signal()
    redo_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.target_label = QLabel("Select a file (1) or a mesh in the group list (2) to transform it.")
        self.target_label.setWordWrap(True)
        self.target_label.setObjectName("sub")
        v.addWidget(self.target_label)

        g = QGridLayout()
        g.setHorizontalSpacing(4)
        self.loc = [_spin(-1e7, 1e7, 4, 0.01, "") for _ in range(3)]
        self.rot = [_spin(-36000, 36000, 3, 5.0, "°") for _ in range(3)]
        self.scl = [_spin(-1e6, 1e6, 5, 0.01, "") for _ in range(3)]
        for c, axis in enumerate("XYZ"):
            lab = QLabel(axis)
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet(f"color: {['#e06c6c', '#8fd16a', '#6ca2e0'][c]}; font-weight: 600")
            g.addWidget(lab, 0, c + 1)
        for r, (name, row) in enumerate((("Location", self.loc), ("Rotation", self.rot),
                                         ("Scale", self.scl)), start=1):
            g.addWidget(QLabel(name), r, 0)
            for c, w in enumerate(row):
                g.addWidget(w, r, c + 1)
                w.valueChanged.connect(lambda _v, row=row, c=c: self._edited(row, c))
        self.uniform = QCheckBox("uniform")
        self.uniform.setChecked(True)
        self.uniform.setToolTip("Edit X, Y and Z scale together")
        g.addWidget(self.uniform, 3, 4)
        v.addLayout(g)

        q = QHBoxLayout()
        q.setSpacing(3)
        rotm = QToolButton(text="Rotate 90°")
        rotm.setPopupMode(QToolButton.InstantPopup)
        m = QMenu(rotm)
        for axis in "XYZ":
            for sgn in (90, -90):
                m.addAction(f"{'+' if sgn > 0 else '−'}90° about {axis}",
                            lambda a=axis, s=sgn: self.quick_rotate(a, s))
        m.addSeparator()
        m.addAction("Y-up → Z-up  (+90° about X)", lambda: self.quick_rotate("X", 90))
        m.addAction("Z-up → Y-up  (−90° about X)", lambda: self.quick_rotate("X", -90))
        rotm.setMenu(m)
        q.addWidget(rotm)
        sclm = QToolButton(text="Units")
        sclm.setPopupMode(QToolButton.InstantPopup)
        m2 = QMenu(sclm)
        for f, label, desc in QUICK_SCALES:
            m2.addAction(f"{label}   {desc}".strip(), lambda f=f: self.quick_scale(f))
        sclm.setMenu(m2)
        q.addWidget(sclm)
        self.mirror_btn = QToolButton(text="Mirror")
        mm = QMenu(self.mirror_btn)
        for i, axis in enumerate("XYZ"):
            mm.addAction(f"Mirror {axis}", lambda i=i: self.quick_mirror(i))
        self.mirror_btn.setMenu(mm)
        self.mirror_btn.setPopupMode(QToolButton.InstantPopup)
        q.addWidget(self.mirror_btn)
        reset = QToolButton(text="Reset")
        reset.setToolTip("Clear this transform")
        reset.clicked.connect(self.reset_requested.emit)
        q.addWidget(reset)
        q.addStretch(1)
        self.undo_btn = QToolButton(text="↶")
        self.undo_btn.setToolTip("Undo transform change (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo_requested.emit)
        self.redo_btn = QToolButton(text="↷")
        self.redo_btn.setToolTip("Redo (Ctrl+Shift+Z)")
        self.redo_btn.clicked.connect(self.redo_requested.emit)
        q.addWidget(self.undo_btn)
        q.addWidget(self.redo_btn)
        v.addLayout(q)

        b = QHBoxLayout()
        self.align_btn = QPushButton("Auto-align group")
        self.align_btn.setToolTip("Find the scale, rotation and offset that put this group's high "
                                  "polys onto its low poly")
        self.align_btn.clicked.connect(self.align_requested.emit)
        self.export_btn = QPushButton("Export corrected meshes…")
        self.export_btn.setToolTip("Write copies of the transformed files with the correction "
                                   "applied to the geometry")
        self.export_btn.clicked.connect(self.export_requested.emit)
        b.addWidget(self.align_btn)
        b.addWidget(self.export_btn)
        v.addLayout(b)

        self._t = xform.identity()
        self._quiet = False
        self.set_target(None)

    # ------------------------------------------------------------------ API
    def set_target(self, label, transform=None, can_align=False):
        """label: text shown above the fields, or None to disable editing."""
        enabled = label is not None
        self.target_label.setText(label or "Select a file (section 1) or a mesh in the group "
                                           "list to transform it.")
        for w in self.loc + self.rot + self.scl + [self.uniform, self.mirror_btn]:
            w.setEnabled(enabled)
        self.align_btn.setEnabled(can_align)
        self.show_transform(transform or xform.identity())

    def show_transform(self, t):
        self._t = xform.normalized(t)
        self._quiet = True
        for row, key in ((self.loc, "loc"), (self.rot, "rot"), (self.scl, "scale")):
            for w, val in zip(row, self._t[key]):
                w.setValue(val)
        s = self._t["scale"]
        if abs(s[0] - s[1]) > 1e-9 or abs(s[0] - s[2]) > 1e-9:
            self.uniform.setChecked(False)
        self._quiet = False

    def set_undo_state(self, can_undo, can_redo):
        self.undo_btn.setEnabled(can_undo)
        self.redo_btn.setEnabled(can_redo)

    def transform(self):
        return xform.normalized(self._t)

    # ------------------------------------------------------------------ edits
    def _edited(self, row, c):
        if self._quiet:
            return
        if row is self.scl and self.uniform.isChecked():
            val = self.scl[c].value()
            self._quiet = True
            for w in self.scl:
                w.setValue(val)
            self._quiet = False
        self._t = {"loc": [w.value() for w in self.loc], "rot": [w.value() for w in self.rot],
                   "scale": [w.value() for w in self.scl]}
        self.changed.emit(self.transform())

    def _apply_world(self, M):
        """Pre-multiply a world-space matrix onto the current transform."""
        self.show_transform(xform.decompose(M @ xform.matrix(self._t)))
        self.changed.emit(self.transform())

    def quick_rotate(self, axis, deg):
        r = [0.0, 0.0, 0.0]
        r["XYZ".index(axis)] = deg
        self._apply_world(xform.matrix({"rot": r}))

    def quick_scale(self, f):
        self._apply_world(xform.matrix({"scale": [f, f, f]}))

    def quick_mirror(self, i):
        s = [1.0, 1.0, 1.0]
        s[i] = -1.0
        self.uniform.setChecked(False)
        self._apply_world(xform.matrix({"scale": s}))
