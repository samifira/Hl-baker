# SPDX-License-Identifier: GPL-3.0-or-later
"""Project / settings model. Pure Python - no bpy import, safe for the GUI process."""
import json
import re
from dataclasses import dataclass, field, asdict, fields

APP_NAME = "HL Baker"
APP_VERSION = "1.2.0"
PROJECT_EXT = ".hlbake"

# key: (label, file suffix, is_color(sRGB), background RGB)
MAPS = {
    "normal":      ("Normal (Tangent)",  "normal",      False, (0.5, 0.5, 1.0)),
    "normal_obj":  ("Normal (Object)",   "normal_obj",  False, (0.5, 0.5, 0.5)),
    "ao":          ("Ambient Occlusion", "ao",          False, (1.0, 1.0, 1.0)),
    "curvature":   ("Curvature",         "curvature",   False, (0.5, 0.5, 0.5)),
    "thickness":   ("Thickness",         "thickness",   False, (0.0, 0.0, 0.0)),
    "height":      ("Height",            "height",      False, (0.5, 0.5, 0.5)),
    "position":    ("Position",          "position",    False, (0.0, 0.0, 0.0)),
    "material_id": ("Material ID",       "matid",       True,  (0.0, 0.0, 0.0)),
    "object_id":   ("Object ID",         "objid",       True,  (0.0, 0.0, 0.0)),
    "albedo":      ("Albedo",            "albedo",      True,  (0.5, 0.5, 0.5)),
    "roughness":   ("Roughness",         "roughness",   False, (0.5, 0.5, 0.5)),
    "metalness":   ("Metalness",         "metalness",   False, (0.0, 0.0, 0.0)),
    "emissive":    ("Emissive",          "emissive",    True,  (0.0, 0.0, 0.0)),
    "opacity":     ("Opacity",           "opacity",     False, (1.0, 1.0, 1.0)),
}
MAP_ORDER = list(MAPS.keys())
MATERIAL_MAPS = ("albedo", "roughness", "metalness", "emissive", "opacity")
GAME_PRESET = ("normal", "ao", "curvature", "thickness", "material_id", "position")

FORMATS = {
    # key: (blender file_format, color_depth, extension, is_float_file)
    "PNG8":   ("PNG", "8", "png", False),
    "PNG16":  ("PNG", "16", "png", False),
    "TGA":    ("TARGA", "8", "tga", False),
    "TIFF16": ("TIFF", "16", "tif", False),
    "EXR32":  ("OPEN_EXR", "32", "exr", True),
}
FORMAT_LABELS = {"PNG8": "PNG 8-bit", "PNG16": "PNG 16-bit", "TGA": "Targa 8-bit",
                 "TIFF16": "TIFF 16-bit", "EXR32": "OpenEXR 32-bit float"}

MESH_EXTS = (".fbx", ".obj", ".gltf", ".glb", ".ply", ".stl", ".abc", ".usd", ".usda", ".usdc",
             ".usdz", ".blend")

SUFFIX_RE = re.compile(
    r"^(?P<base>.+?)[ _.\-]+(?P<kind>low|lp|lo|high|hp|hi|cage)(?:[ _.\-]*\d+)?$",
    re.IGNORECASE)
KIND_MAP = {"low": "low", "lp": "low", "lo": "low",
            "high": "high", "hp": "high", "hi": "high", "cage": "cage"}
DUP_RE = re.compile(r"\.\d{3}$")


def split_name(name):
    """'crate_high.001' -> ('crate', 'high');  'crate' -> ('crate', None)."""
    name = DUP_RE.sub("", name)
    m = SUFFIX_RE.match(name)
    if m:
        return m.group("base"), KIND_MAP[m.group("kind").lower()]
    return name, None


def guess_file_role(path):
    import os
    stem = os.path.splitext(os.path.basename(path))[0]
    _, kind = split_name(stem)
    return kind or "auto"


@dataclass
class Settings:
    # output
    output_dir: str = ""
    base_name: str = "bake"
    output_mode: str = "SINGLE"        # SINGLE | PER_GROUP
    width: int = 2048
    height: int = 2048
    supersample: str = "2"             # '1' | '2' | '4'
    ray_samples: int = 64
    padding: int = 16
    padding_full: bool = False
    file_format: str = "PNG16"
    device: str = "GPU"
    use_uv: str = ""
    # maps
    map_normal: bool = True
    map_normal_obj: bool = False
    map_ao: bool = True
    map_curvature: bool = True
    map_thickness: bool = False
    map_height: bool = False
    map_position: bool = False
    map_material_id: bool = False
    map_object_id: bool = False
    map_albedo: bool = False
    map_roughness: bool = False
    map_metalness: bool = False
    map_emissive: bool = False
    map_opacity: bool = False
    # map options
    normal_flip_y: str = "OPENGL"      # OPENGL | DIRECTX
    ao_distance: float = 0.5
    ao_ignore_groups: bool = False
    ao_only_self: bool = False
    curv_method: str = "RAY"           # RAY | POINTINESS
    curv_radius: float = 0.01
    curv_strength: float = 3.0
    thick_distance: float = 0.2
    height_auto: bool = True
    height_inner: float = 0.05
    height_outer: float = 0.05
    position_space: str = "BOUNDS"     # BOUNDS | WORLD
    matid_source: str = "HASH"         # HASH | VIEWPORT


@dataclass
class FileEntry:
    path: str
    role: str = "auto"                 # auto | low | high | cage
    scale: float = 1.0                 # import scale (CLI --scale)
    transform: dict = field(default_factory=lambda: {"loc": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0],
                                                     "scale": [1.0, 1.0, 1.0]})


@dataclass
class GroupConfig:
    name: str
    low: str
    highs: list = field(default_factory=list)
    cage: str = ""
    cage_offset: float = 0.05
    max_ray: float = 0.0
    enabled: bool = True


@dataclass
class Project:
    files: list = field(default_factory=list)       # [FileEntry]
    groups: list = field(default_factory=list)      # [GroupConfig]; empty = auto
    settings: Settings = field(default_factory=Settings)
    default_cage_offset: float = 0.05
    default_max_ray: float = 0.0
    object_transforms: dict = field(default_factory=dict)   # object name -> transform (on top of file)
    hidden: list = field(default_factory=list)              # object names hidden in the 3D viewer

    def to_dict(self):
        return {"app": APP_NAME, "version": APP_VERSION,
                "files": [asdict(f) for f in self.files],
                "groups": [asdict(g) for g in self.groups],
                "settings": asdict(self.settings),
                "default_cage_offset": self.default_cage_offset,
                "default_max_ray": self.default_max_ray,
                "object_transforms": self.object_transforms,
                "hidden": list(self.hidden)}

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(Settings)}
        st = Settings(**{k: v for k, v in d.get("settings", {}).items() if k in known})
        fe_known = {f.name for f in fields(FileEntry)}
        return cls(files=[FileEntry(**{k: v for k, v in f.items() if k in fe_known})
                          for f in d.get("files", [])],
                   groups=[GroupConfig(**g) for g in d.get("groups", [])],
                   settings=st,
                   default_cage_offset=d.get("default_cage_offset", 0.05),
                   default_max_ray=d.get("default_max_ray", 0.0),
                   object_transforms=dict(d.get("object_transforms", {})),
                   hidden=list(d.get("hidden", [])))

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def enabled_maps(s):
    return [k for k in MAP_ORDER if getattr(s, "map_" + k)]


def clean_name(name):
    """Same as Blender's bpy.path.clean_name (texture set names must match the worker's)."""
    return re.sub(r"[^A-Za-z0-9]", "_", name)


def texture_set_name(settings, group_name=None):
    """Texture set a group's maps are written to (mirrors core.texture_set_name)."""
    if settings.output_mode == "PER_GROUP" and group_name is not None:
        return clean_name(group_name)
    return clean_name(settings.base_name.strip()) if settings.base_name.strip() else "bake"
