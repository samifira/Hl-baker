# SPDX-License-Identifier: GPL-3.0-or-later
"""Command line interface and the JSON worker used by the GUI."""
import argparse
import json
import os
import sys
import time

from .config import (Project, FileEntry, Settings, MAP_ORDER, MAPS, FORMATS, GAME_PRESET,
                     guess_file_role, APP_NAME, APP_VERSION)

EVENT_PREFIX = "@@HLB "


def _emit(**ev):
    sys.stdout.write(EVENT_PREFIX + json.dumps(ev) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Worker (GUI <-> engine). One command per process.
# ---------------------------------------------------------------------------
def run_worker(argv):
    ap = argparse.ArgumentParser(prog="hlbake worker")
    ap.add_argument("command", choices=["scan", "bake", "render"])
    ap.add_argument("project")
    ap.add_argument("--out", default="")
    ap.add_argument("--yaw", type=float, default=35.0)
    ap.add_argument("--pitch", type=float, default=25.0)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--show", default="low")
    ap.add_argument("--preview-dir", default="")
    ap.add_argument("--mesh-out", default="")
    a = ap.parse_args(argv)
    try:
        project = Project.load(a.project)
        from . import scene
        if a.command == "scan":
            _emit(event="scan", **scene.scan(project, mesh_out=a.mesh_out or None))
        elif a.command == "bake":
            t0 = time.time()
            written = scene.bake(project, _emit, preview_dir=a.preview_dir or None)
            _emit(event="done", files=written, seconds=round(time.time() - t0, 1))
        elif a.command == "render":
            path = scene.render_preview(project, a.out, yaw=a.yaw, pitch=a.pitch,
                                        samples=a.samples, show=a.show)
            _emit(event="render", path=path)
        return 0
    except Exception as e:  # report every failure to the GUI
        import traceback
        _emit(event="error", message=str(e), trace=traceback.format_exc())
        return 1


# ---------------------------------------------------------------------------
# Human CLI
# ---------------------------------------------------------------------------
def _project_from_args(a):
    if a.project:
        p = Project.load(a.project)
    else:
        p = Project()
    for role in ("low", "high", "cage", "auto"):
        for f in getattr(a, role) or []:
            p.files.append(FileEntry(os.path.abspath(f), role, a.scale))
    for f in a.files or []:
        p.files.append(FileEntry(os.path.abspath(f), guess_file_role(f), a.scale))
    s = p.settings
    if a.out:
        s.output_dir = os.path.abspath(a.out)
    if not s.output_dir:
        s.output_dir = os.path.abspath("bakes")
    if a.name:
        s.base_name = a.name
    if a.size:
        s.width = s.height = a.size
    if a.width:
        s.width = a.width
    if a.height:
        s.height = a.height
    if a.aa:
        s.supersample = str(a.aa)
    if a.samples:
        s.ray_samples = a.samples
    if a.padding is not None:
        s.padding = a.padding
    if a.fill:
        s.padding_full = True
    if a.format:
        s.file_format = a.format
    if a.device:
        s.device = a.device
    if a.per_group:
        s.output_mode = "PER_GROUP"
    if a.directx:
        s.normal_flip_y = "DIRECTX"
    if a.maps:
        want = MAP_ORDER if a.maps == "all" else GAME_PRESET if a.maps == "game" else \
            [m.strip() for m in a.maps.split(",") if m.strip()]
        bad = [m for m in want if m not in MAPS]
        if bad:
            raise SystemExit(f"Unknown map(s): {', '.join(bad)}. Choose from: {', '.join(MAP_ORDER)}")
        for k in MAP_ORDER:
            setattr(s, "map_" + k, k in want)
    if a.cage_offset is not None:
        p.default_cage_offset = a.cage_offset
        for g in p.groups:
            g.cage_offset = a.cage_offset
    if a.max_ray is not None:
        p.default_max_ray = a.max_ray
        for g in p.groups:
            g.max_ray = a.max_ray
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        if not hasattr(s, k):
            raise SystemExit(f"Unknown setting: {k}")
        cur = getattr(s, k)
        setattr(s, k, (v.lower() in ("1", "true", "yes", "on")) if isinstance(cur, bool)
                else type(cur)(v))
    return p


def _add_common(ap):
    ap.add_argument("files", nargs="*", help="Mesh files. Role guessed from file name (_low/_high/_cage)")
    ap.add_argument("--project", "-p", help=".hlbake project file")
    ap.add_argument("--low", "-l", action="append", help="File whose meshes are all low polys")
    ap.add_argument("--high", "-H", action="append", help="File whose meshes are all high polys")
    ap.add_argument("--cage", action="append", help="File containing cage meshes")
    ap.add_argument("--auto", action="append", help="File with mixed meshes named *_low / *_high")
    ap.add_argument("--scale", type=float, default=1.0, help="Import scale for files given here")
    ap.add_argument("--cage-offset", type=float)
    ap.add_argument("--max-ray", type=float)


def main_cli(argv):
    ap = argparse.ArgumentParser(prog="hlbake", description=f"{APP_NAME} {APP_VERSION} - "
                                 "high-to-low poly texture baker (Cycles)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bake", help="Bake maps")
    _add_common(b)
    b.add_argument("--out", "-o", help="Output folder (default ./bakes)")
    b.add_argument("--name", "-n", help="Texture set name (default 'bake')")
    b.add_argument("--maps", "-m", help=f"Comma list, 'game' or 'all'. Maps: {', '.join(MAP_ORDER)}")
    b.add_argument("--size", "-s", type=int, help="Square resolution, e.g. 2048")
    b.add_argument("--width", type=int)
    b.add_argument("--height", type=int)
    b.add_argument("--aa", type=int, choices=[1, 2, 4], help="Supersampling factor per axis")
    b.add_argument("--samples", type=int, help="Ray samples for AO/thickness/curvature")
    b.add_argument("--padding", type=int)
    b.add_argument("--fill", action="store_true", help="Pad until the whole texture is filled")
    b.add_argument("--format", "-f", choices=list(FORMATS))
    b.add_argument("--device", choices=["CPU", "GPU"])
    b.add_argument("--per-group", action="store_true", help="Write separate textures per group")
    b.add_argument("--directx", action="store_true", help="DirectX (Y-) normal maps")
    b.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="Any project setting, e.g. --set ao_distance=1.5")
    b.add_argument("--save-project", help="Also write the resulting .hlbake project")

    sc = sub.add_parser("scan", help="Import files and show the bake groups that would be used")
    _add_common(sc)

    a = ap.parse_args(argv)
    if a.cmd == "scan":
        for k in ("out", "name", "size", "width", "height", "aa", "samples", "padding", "fill",
                  "format", "device", "per_group", "directx", "maps", "set"):
            setattr(a, k, None)
    p = _project_from_args(a)
    if not p.files:
        ap.error("no input files")

    from . import scene
    if a.cmd == "scan":
        r = scene.scan(p)
        print(f"\n{len(r['objects'])} mesh object(s):")
        for o in r["objects"]:
            print(f"  [{o['role']:>10}] {o['name']:<32} {o['tris']:>9,} tris  UVs: {', '.join(o['uvs']) or '-'}")
        print(f"\n{len(r['groups'])} bake group(s):")
        for g in r["groups"]:
            print(f"  {g['name']}: low={g['low']}  high={', '.join(g['highs']) or '-'}"
                  + (f"  cage={g['cage']}" if g['cage'] else ""))
        if r["unmatched"]:
            print("\nLow polys without a high poly:", ", ".join(r["unmatched"]))
        return 0

    if getattr(a, "save_project", None):
        p.save(a.save_project)

    t0 = time.time()

    def progress(**ev):
        e = ev.get("event")
        if e == "progress":
            n, t = ev["step"], ev["total"]
            bar = "#" * int(30 * n / t)
            sys.stderr.write(f"\r[{bar:<30}] {n}/{t}  {ev['map']} / {ev['group']}          ")
            sys.stderr.flush()
        elif e == "log":
            sys.stderr.write("\n" + ev["message"] + "\n")
        elif e == "map_done":
            sys.stderr.write(f"\n  wrote {ev['path']}")

    try:
        files = scene.bake(p, progress)
    except scene.SceneError as e:
        sys.stderr.write(f"\nerror: {e}\n")
        return 2
    sys.stderr.write(f"\n\nDone: {len(files)} file(s) in {time.time() - t0:.1f}s -> "
                     f"{p.settings.output_dir}\n")
    return 0
