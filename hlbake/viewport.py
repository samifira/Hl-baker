# SPDX-License-Identifier: GPL-3.0-or-later
"""Live OpenGL viewer: the low poly shaded with the baked maps, plus high poly / cage / wireframe.

Runs in the GUI process (no bpy). Meshes come from the worker as an .npz file
(see meshexport.py); textures are the PNG previews the worker writes after each map.
Requires OpenGL 3.3 core (4.1 on macOS) via PyOpenGL.
"""
import ctypes
import json
import math
import os

import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QImage, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

try:
    import OpenGL           # noqa: F401  (OpenGL.GL is imported lazily, see _load_gl)
    HAVE_GL = True
except Exception:           # PyOpenGL missing
    HAVE_GL = False
GL = None
GL_DIAG = {}


def _load_gl():
    """Import PyOpenGL once a Qt context exists.

    When Qt runs on its bundled software renderer (Windows, opengl32sw.dll), PyOpenGL must call
    into that same DLL instead of the system opengl32.dll, or its calls see no current context.
    """
    global GL
    if GL is not None:
        return GL
    import sys
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetModuleHandleW.restype = ctypes.c_void_p
        k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        handle = k32.GetModuleHandleW("opengl32sw.dll")
        GL_DIAG["opengl32sw_loaded"] = bool(handle)
        GL_DIAG["opengl32_loaded"] = bool(k32.GetModuleHandleW("opengl32.dll"))
        if handle and os.environ.get("HLBAKE_SOFTWARE_GL") == "1":
            sw = ctypes.WinDLL("opengl32sw.dll", handle=handle)
            from OpenGL.platform import ctypesloader
            _orig = ctypesloader.loadLibrary

            def _load(dll_type, name, mode=ctypes.RTLD_GLOBAL):
                if str(name).lower() == "opengl32":
                    return sw
                return _orig(dll_type, name, mode)
            ctypesloader.loadLibrary = _load
    from OpenGL import GL as _GL
    GL = _GL
    return GL

# viewer channel -> (map key used, label)
CHANNELS = [
    ("lit", "Lit (all maps)"),
    ("normal", "Normal (tangent)"),
    ("normal_obj", "Normal (object)"),
    ("ao", "Ambient Occlusion"),
    ("curvature", "Curvature"),
    ("thickness", "Thickness"),
    ("height", "Height"),
    ("position", "Position"),
    ("material_id", "Material ID"),
    ("object_id", "Object ID"),
    ("albedo", "Albedo"),
    ("roughness", "Roughness"),
    ("metalness", "Metalness"),
    ("emissive", "Emissive"),
    ("opacity", "Opacity"),
    ("uv", "UV checker"),
    ("geo_normal", "Mesh normals"),
]
LIT_MAPS = ("albedo", "normal", "ao", "roughness", "metalness", "emissive")
DEFAULTS = {  # value used when a map hasn't been baked (RGBA, 0-255)
    "albedo": (150, 150, 150, 255), "normal": (128, 128, 255, 255), "ao": (255, 255, 255, 255),
    "roughness": (140, 140, 140, 255), "metalness": (0, 0, 0, 255), "emissive": (0, 0, 0, 255),
}


def request_gl_format():
    """Call before QApplication is created."""
    f = QSurfaceFormat()
    f.setVersion(3, 3)
    f.setProfile(QSurfaceFormat.CoreProfile)
    f.setDepthBufferSize(24)
    f.setSamples(4)
    QSurfaceFormat.setDefaultFormat(f)


# ---------------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------------
LOW_VS = """#version 330 core
layout(location=0) in vec3 aP; layout(location=1) in vec3 aN; layout(location=2) in vec3 aT;
layout(location=3) in float aS; layout(location=4) in vec2 aUV; layout(location=5) in vec3 aVN;
layout(location=6) in vec3 aBary;
uniform mat4 uVP; uniform float uOffset;
out vec3 vP; out vec3 vN; out vec3 vT; out float vS; out vec2 vUV; out vec3 vBary;
void main(){
  vec3 p = aP + aVN * uOffset;
  vP = p; vN = aN; vT = aT; vS = aS; vUV = aUV; vBary = aBary;
  gl_Position = uVP * vec4(p, 1.0);
}"""

LOW_FS = """#version 330 core
in vec3 vP; in vec3 vN; in vec3 vT; in float vS; in vec2 vUV; in vec3 vBary;
uniform sampler2D tAlbedo, tNormal, tAO, tRough, tMetal, tEmit, tChannel;
uniform int uMode;          // 0 lit, 1 show texture, 2 uv checker, 3 geometry normals
uniform int uUseNormalMap; uniform int uDirectX; uniform int uUseAO; uniform int uWire;
uniform vec3 uCam; uniform vec3 uLightDir; uniform float uExposure; uniform vec3 uWireCol;
out vec4 frag;
const float PI = 3.14159265;
vec3 srgb2lin(vec3 c){ return mix(c/12.92, pow((c+0.055)/1.055, vec3(2.4)), step(0.04045, c)); }
vec3 lin2srgb(vec3 c){ c = max(c, 0.0); return mix(c*12.92, 1.055*pow(c, vec3(1.0/2.4))-0.055, step(0.0031308, c)); }
vec3 sky(vec3 d){                       // simple studio environment
  float y = d.z;
  vec3 top = vec3(0.70, 0.74, 0.80), hor = vec3(0.45, 0.45, 0.47), bot = vec3(0.11, 0.11, 0.12);
  vec3 c = y > 0.0 ? mix(hor, top, pow(y, 0.6)) : mix(hor, bot, pow(-y, 0.4));
  float soft = smoothstep(0.75, 1.0, dot(d, normalize(vec3(-0.5, -0.6, 0.62))));
  return c + soft * vec3(1.6, 1.55, 1.5);
}
vec3 aces(vec3 x){ return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), 0.0, 1.0); }
float wire(){ vec3 d = fwidth(vBary); vec3 a = smoothstep(vec3(0.0), d*1.2, vBary); return 1.0 - min(min(a.x,a.y),a.z); }
void main(){
  vec3 Ng = normalize(vN);
  if (!gl_FrontFacing) Ng = -Ng;
  vec3 col;
  if (uMode == 1) {
    col = texture(tChannel, vUV).rgb;                    // displayed as stored, like an image viewer
  } else if (uMode == 2) {
    vec2 c = floor(vUV * 16.0); float k = mod(c.x + c.y, 2.0);
    col = mix(vec3(0.25), vec3(0.85), k) * mix(vec3(1.0,0.55,0.25), vec3(0.35,0.6,1.0), vUV.x);
    col = mix(col, vec3(0.9), 0.35*step(0.98, fract(vUV.x*16.0)) );
  } else if (uMode == 3) {
    col = Ng * 0.5 + 0.5;
  } else {
    vec3 base = srgb2lin(texture(tAlbedo, vUV).rgb);
    float rough = clamp(texture(tRough, vUV).r, 0.04, 1.0);
    float metal = texture(tMetal, vUV).r;
    float ao = uUseAO == 1 ? texture(tAO, vUV).r : 1.0;
    vec3 emit = srgb2lin(texture(tEmit, vUV).rgb);
    vec3 N = Ng;
    if (uUseNormalMap == 1) {
      vec3 tn = texture(tNormal, vUV).rgb * 2.0 - 1.0;
      if (uDirectX == 1) tn.y = -tn.y;
      vec3 T = vT - Ng * dot(Ng, vT);                    // MikkTSpace-style per pixel frame
      vec3 B = vS * cross(vN, vT);
      N = normalize(tn.x * T + tn.y * B + tn.z * Ng);
    }
    vec3 V = normalize(uCam - vP);
    float NoV = max(dot(N, V), 1e-4);
    vec3 F0 = mix(vec3(0.04), base, metal);
    vec3 diff = base * (1.0 - metal);
    vec3 Lo = vec3(0.0);
    vec3 Ls[2] = vec3[2](normalize(uLightDir), normalize(vec3(-uLightDir.x, -uLightDir.y, 0.35)));
    vec3 Lc[2] = vec3[2](vec3(3.2, 3.05, 2.9), vec3(0.9, 1.0, 1.15));
    float a2 = rough*rough*rough*rough;
    for (int i = 0; i < 2; i++) {
      vec3 L = Ls[i]; vec3 H = normalize(L + V);
      float NoL = max(dot(N, L), 0.0), NoH = max(dot(N, H), 0.0), VoH = max(dot(V, H), 0.0);
      float D = a2 / (PI * pow(NoH*NoH*(a2-1.0)+1.0, 2.0));
      float k = (rough+1.0)*(rough+1.0)/8.0;
      float G = (NoV/(NoV*(1.0-k)+k)) * (NoL/(NoL*(1.0-k)+k));
      vec3 F = F0 + (1.0-F0) * pow(1.0-VoH, 5.0);
      vec3 spec = D * G * F / max(4.0*NoV*NoL, 1e-4);
      Lo += (diff/PI * (1.0-F) + spec) * Lc[i] * NoL;
    }
    // ambient: irradiance + roughness-blurred reflection from the studio sky
    vec3 R = reflect(-V, N);
    vec3 irr = 0.5*(sky(N) + sky(normalize(N + vec3(0,0,0.6))));
    vec3 refl = mix(sky(R), irr, rough*rough);
    vec3 Fa = F0 + (max(vec3(1.0-rough), F0) - F0) * pow(1.0-NoV, 5.0);
    vec3 amb = (diff * irr * (1.0 - Fa) + refl * Fa) * ao;
    col = (Lo * mix(1.0, ao, 0.35) + amb * 0.85 + emit) * uExposure;
    col = lin2srgb(aces(col));
  }
  if (uWire == 1) col = mix(col, uWireCol, wire() * 0.85);
  frag = vec4(col, 1.0);
}"""

SOLID_VS = """#version 330 core
layout(location=0) in vec3 aP; layout(location=1) in vec3 aN;
uniform mat4 uVP; out vec3 vN; out vec3 vP;
void main(){ vN = aN; vP = aP; gl_Position = uVP * vec4(aP, 1.0); }"""

SOLID_FS = """#version 330 core
in vec3 vN; in vec3 vP; uniform vec3 uCam; uniform vec3 uLightDir; uniform vec3 uColor;
uniform float uAlpha; out vec4 frag;
void main(){
  vec3 N = normalize(vN); vec3 V = normalize(uCam - vP);
  if (dot(N, V) < 0.0) N = -N;
  float d = max(dot(N, normalize(uLightDir)), 0.0);
  float h = 0.5 + 0.5*N.z; float rim = pow(1.0 - max(dot(N,V),0.0), 3.0);
  vec3 c = uColor * (0.25 + 0.55*d + 0.25*h) + rim*0.15;
  frag = vec4(pow(c, vec3(1.0/1.1)), uAlpha);
}"""

LINE_VS = """#version 330 core
layout(location=0) in vec3 aP; layout(location=5) in vec3 aVN;
uniform mat4 uVP; uniform float uOffset;
void main(){ gl_Position = uVP * vec4(aP + aVN*uOffset, 1.0); }"""

LINE_FS = """#version 330 core
uniform vec4 uColor; out vec4 frag; void main(){ frag = uColor; }"""

BG_VS = """#version 330 core
out vec2 vUV; void main(){ vec2 p = vec2((gl_VertexID<<1)&2, gl_VertexID&2);
vUV = p; gl_Position = vec4(p*2.0-1.0, 0.999, 1.0); }"""

BG_FS = """#version 330 core
in vec2 vUV; out vec4 frag;
void main(){ vec3 top = vec3(0.20,0.21,0.235), bot = vec3(0.105,0.11,0.12);
float v = smoothstep(0.0, 1.0, vUV.y); vec2 c = vUV - 0.5;
frag = vec4(mix(bot, top, v) * (1.0 - 0.35*dot(c,c)), 1.0); }"""

GRID_FS = LINE_FS


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(fovy / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def look_at(eye, target, up=(0, 0, 1)):
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-6:
        s = np.cross(f, (0, 1, 0))
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4, np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -s @ eye, -u @ eye, f @ eye
    return m


# ---------------------------------------------------------------------------
# GL objects
# ---------------------------------------------------------------------------
def _compile(vs, fs):
    def sh(src, kind):
        s = GL.glCreateShader(kind)
        GL.glShaderSource(s, src)
        GL.glCompileShader(s)
        if not GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(s).decode())
        return s
    p = GL.glCreateProgram()
    a, b = sh(vs, GL.GL_VERTEX_SHADER), sh(fs, GL.GL_FRAGMENT_SHADER)
    GL.glAttachShader(p, a)
    GL.glAttachShader(p, b)
    GL.glLinkProgram(p)
    if not GL.glGetProgramiv(p, GL.GL_LINK_STATUS):
        raise RuntimeError(GL.glGetProgramInfoLog(p).decode())
    GL.glDeleteShader(a)
    GL.glDeleteShader(b)
    return p


class Program:
    def __init__(self, vs, fs):
        self.id = _compile(vs, fs)
        self._loc = {}

    def use(self):
        GL.glUseProgram(self.id)

    def loc(self, name):
        if name not in self._loc:
            self._loc[name] = GL.glGetUniformLocation(self.id, name)
        return self._loc[name]

    def set(self, name, v):
        l = self.loc(name)
        if l < 0:
            return
        if isinstance(v, (bool, int)) and not isinstance(v, float):
            GL.glUniform1i(l, int(v))
        elif isinstance(v, float):
            GL.glUniform1f(l, v)
        else:
            a = np.asarray(v, np.float32)
            if a.shape == (4, 4):
                GL.glUniformMatrix4fv(l, 1, GL.GL_TRUE, a)
            elif a.shape == (3,):
                GL.glUniform3f(l, *a)
            elif a.shape == (4,):
                GL.glUniform4f(l, *a)


class Mesh:
    """VAO with float attributes at given locations; optional index buffer."""

    def __init__(self, attribs, indices=None):
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        self.vbos = []
        n = None
        for loc, arr in attribs.items():
            arr = np.ascontiguousarray(arr, np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            n = len(arr)
            vbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, arr.nbytes, arr, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(loc)
            GL.glVertexAttribPointer(loc, arr.shape[1], GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            self.vbos.append(vbo)
        self.count = n
        self.ebo = None
        self.line_ebo = None
        if indices is not None:
            idx = np.ascontiguousarray(indices, np.uint32).ravel()
            self.ebo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)
            self.count = len(idx)
        GL.glBindVertexArray(0)

    def set_lines(self, line_idx):
        """Index buffer of unique edges, drawn with GL_LINES (cage)."""
        GL.glBindVertexArray(self.vao)
        idx = np.ascontiguousarray(line_idx, np.uint32).ravel()
        self.line_ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.line_ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)
        self.line_count = len(idx)
        GL.glBindVertexArray(0)

    def draw(self):
        GL.glBindVertexArray(self.vao)
        if self.ebo is not None:
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            GL.glDrawElements(GL.GL_TRIANGLES, self.count, GL.GL_UNSIGNED_INT, None)
        else:
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.count)

    def draw_lines(self):
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.line_ebo)
        GL.glDrawElements(GL.GL_LINES, self.line_count, GL.GL_UNSIGNED_INT, None)

    def delete(self):
        GL.glDeleteBuffers(len(self.vbos), self.vbos)
        for b in (self.ebo, self.line_ebo):
            if b is not None:
                GL.glDeleteBuffers(1, [b])
        GL.glDeleteVertexArrays(1, [self.vao])


def _upload_texture(img: QImage, tex=None):
    img = img.convertToFormat(QImage.Format_RGBA8888).mirrored(False, True)  # GL origin bottom-left
    w, h = img.width(), img.height()
    data = bytes(img.constBits())[: w * h * 4]
    if tex is None:
        tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    try:
        GL.glTexParameterf(GL.GL_TEXTURE_2D, 0x84FE, 8.0)    # anisotropic filtering if supported
    except Exception:
        pass
    return tex


def _solid_texture(rgba):
    img = QImage(1, 1, QImage.Format_RGBA8888)
    img.setPixelColor(0, 0, __import__("PySide6.QtGui", fromlist=["QColor"]).QColor(*rgba))
    return _upload_texture(img)


# ---------------------------------------------------------------------------
# The widget
# ---------------------------------------------------------------------------
class Viewport(QOpenGLWidget):
    status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.ok = False
        self.software = False
        self.renderer = ""
        self.error = "" if HAVE_GL else "PyOpenGL is not installed"
        # camera
        self.target = np.zeros(3, np.float32)
        self.yaw, self.pitch, self.dist = 35.0, 22.0, 5.0
        self.radius = 1.0
        self._drag = None
        # display state
        self.channel = "lit"
        self.use_normal_map = True
        self.use_ao = True
        self.directx = False
        self.show_low = True
        self.show_high = False
        self.show_cage = False
        self.show_wire = False
        self.light_yaw = 40.0
        self.exposure = 1.0
        # data
        self._pending_meshes = None
        self._pending_tex = {}          # (set, key) -> path
        self.groups = []                # dicts: name, set, low Mesh, highs [Mesh], cage_offset
        self.cage_offsets = {}
        self.textures = {}              # (set, key) -> GL id
        self.tex_paths = {}             # (set, key) -> path loaded
        self.defaults = {}

    # ------------------------------------------------------------- public API
    def load_meshes(self, path):
        """Queue an .npz from the worker; uploaded on the next paint."""
        self._pending_meshes = path
        self.update()

    def set_texture(self, setname, key, path):
        self._pending_tex[(setname, key)] = path
        self.update()

    def clear_textures(self):
        self._pending_tex[("*", "*")] = None
        self.update()

    def set_cage_offset(self, group_name, value):
        self.cage_offsets[group_name] = float(value)
        self.update()

    def frame(self):
        self.yaw, self.pitch = 35.0, 22.0
        self.dist = self.radius / math.sin(math.radians(22)) * 1.1
        self.update()

    def has_texture(self, setname, key):
        return (setname, key) in self.tex_paths or ((setname, key) in self._pending_tex)

    # ------------------------------------------------------------- GL lifecycle
    def initializeGL(self):
        if not HAVE_GL:
            return
        try:
            ctx = self.context()
            fmt = ctx.format() if ctx else None
            GL_DIAG["qt_ctx_valid"] = bool(ctx and ctx.isValid())
            GL_DIAG["qt_format"] = f"{fmt.majorVersion()}.{fmt.minorVersion()}" if fmt else None
            try:
                GL_DIAG["qt_is_software"] = bool(ctx and ctx.isOpenGLES()) and "gles" or \
                    __import__("PySide6.QtGui", fromlist=["QOpenGLContext"]).QOpenGLContext.openGLModuleType().name
            except Exception as e:
                GL_DIAG["qt_module_err"] = str(e)
            try:
                f = ctx.functions()
                GL_DIAG["qt_glversion"] = str(f.glGetString(0x1F02))
                GL_DIAG["qt_renderer"] = str(f.glGetString(0x1F01))
            except Exception as e:
                GL_DIAG["qt_funcs_err"] = str(e)
            _load_gl()
            ver = GL.glGetString(GL.GL_VERSION)
            if not ver:
                raise RuntimeError("no OpenGL context")
            self.p_low = Program(LOW_VS, LOW_FS)
            self.p_solid = Program(SOLID_VS, SOLID_FS)
            self.p_line = Program(LINE_VS, LINE_FS)
            self.p_bg = Program(BG_VS, BG_FS)
            self.bg_vao = GL.glGenVertexArrays(1)
            for k, rgba in DEFAULTS.items():
                self.defaults[k] = _solid_texture(rgba)
            self.defaults["_black"] = _solid_texture((0, 0, 0, 255))
            self._build_grid()
            self.renderer = GL.glGetString(GL.GL_RENDERER).decode(errors="replace")
            self.ok = True
            import os as _os
            self.software = _os.environ.get("HLBAKE_SOFTWARE_GL") == "1"
            self.status.emit(f"OpenGL {ver.decode(errors='replace').split(' ')[0]} — "
                             f"{GL.glGetString(GL.GL_RENDERER).decode(errors='replace')}")
        except Exception as e:
            self.ok = False
            self.error = f"OpenGL unavailable: {e}"
            self.status.emit(self.error)

    def _build_grid(self):
        n, s = 10, 1.0
        lines = []
        for i in range(-n, n + 1):
            lines += [(i * s, -n * s, 0), (i * s, n * s, 0), (-n * s, i * s, 0), (n * s, i * s, 0)]
        P = np.array(lines, np.float32)
        self.grid = Mesh({0: P, 5: np.zeros_like(P)})
        self.grid.set_lines(np.arange(len(P)))
        self.grid_scale = 1.0

    def _upload_meshes(self, path):
        for g in self.groups:
            g["low"].delete()
            for h in g["highs"]:
                h.delete()
        self.groups = []
        d = np.load(path)
        meta = json.loads(bytes(d["meta"]).decode())
        for gi, gm in enumerate(meta["groups"]):
            P = d[f"g{gi}_low_P"]
            bary = np.tile(np.identity(3, np.float32), (len(P) // 3, 1))
            low = Mesh({0: P, 1: d[f"g{gi}_low_N"], 2: d[f"g{gi}_low_T"], 3: d[f"g{gi}_low_S"],
                        4: d[f"g{gi}_low_UV"], 5: d[f"g{gi}_low_VN"], 6: bary})
            low.set_lines(_unique_edges(P))
            highs = []
            for hi in range(len(gm["highs"])):
                highs.append(Mesh({0: d[f"g{gi}_high{hi}_P"], 1: d[f"g{gi}_high{hi}_N"]},
                                  d[f"g{gi}_high{hi}_F"]))
            self.groups.append({"name": gm["name"], "set": gm["set"], "low": low, "highs": highs})
            self.cage_offsets.setdefault(gm["name"], gm.get("cage_offset", 0.05))
        b = meta.get("bounds")
        if b:
            lo, hi = np.array(b[0], np.float32), np.array(b[1], np.float32)
            self.target = (lo + hi) / 2
            self.radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-3)
            self.grid_scale = 10 ** math.floor(math.log10(self.radius * 2)) / 2
            self.grid_z = float(lo[2])
            self.frame()
        tris = sum(g["low"].count // 3 for g in self.groups)
        htris = sum(h.count // 3 for g in self.groups for h in g["highs"])
        msg = f"{len(self.groups)} group(s) · low {tris:,} tris · high {htris:,} tris"
        if meta.get("high_decimated"):
            msg += " (high simplified for display)"
        self.status.emit(msg)

    def _process_pending(self):
        if self._pending_meshes:
            p, self._pending_meshes = self._pending_meshes, None
            try:
                self._upload_meshes(p)
            except Exception as e:
                self.status.emit(f"Could not load meshes: {e}")
        if self._pending_tex:
            pend, self._pending_tex = self._pending_tex, {}
            if ("*", "*") in pend:
                for tex in self.textures.values():
                    GL.glDeleteTextures(1, [tex])
                self.textures.clear()
                self.tex_paths.clear()
                pend.pop(("*", "*"))
            for key, path in pend.items():
                img = QImage(path) if path and os.path.exists(path) else None
                if img is None or img.isNull():
                    continue
                self.textures[key] = _upload_texture(img, self.textures.get(key))
                self.tex_paths[key] = path

    def _tex(self, setname, key):
        return self.textures.get((setname, key)) or self.defaults.get(key) or self.defaults["_black"]

    def resizeGL(self, w, h):
        pass

    def paintGL(self):
        if not self.ok:
            return
        self._process_pending()
        dpr = self.devicePixelRatioF()
        w, h = int(self.width() * dpr), int(self.height() * dpr)
        GL.glViewport(0, 0, w, h)
        GL.glClearColor(0.12, 0.125, 0.135, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glDisable(GL.GL_DEPTH_TEST)
        self.p_bg.use()
        GL.glBindVertexArray(self.bg_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        try:
            GL.glEnable(GL.GL_MULTISAMPLE)
        except Exception:
            pass

        y, p = math.radians(self.yaw), math.radians(self.pitch)
        d = np.array([math.sin(y) * math.cos(p), -math.cos(y) * math.cos(p), math.sin(p)], np.float32)
        eye = self.target + d * self.dist
        near = max(self.dist * 0.01, self.radius * 0.001)
        VP = perspective(math.radians(40), w / max(h, 1), near, self.dist * 20 + self.radius * 4) @ \
            look_at(eye, self.target)
        ly = math.radians(self.light_yaw + self.yaw)
        light = np.array([math.sin(ly) * 0.7, -math.cos(ly) * 0.7, 0.75], np.float32)

        # ground grid
        if self.groups:
            self.p_line.use()
            s = self.grid_scale
            M = np.diag([s, s, 1, 1]).astype(np.float32)
            M[:3, 3] = (self.target[0], self.target[1], getattr(self, "grid_z", 0.0))
            self.p_line.set("uVP", VP @ M)
            self.p_line.set("uOffset", 0.0)
            self.p_line.set("uColor", (0.32, 0.33, 0.36, 1.0))
            self.grid.draw_lines()

        if self.show_low:
            self.p_low.use()
            pl = self.p_low
            pl.set("uVP", VP)
            pl.set("uCam", eye)
            pl.set("uLightDir", light)
            pl.set("uExposure", float(self.exposure))
            mode = {"lit": 0, "uv": 2, "geo_normal": 3}.get(self.channel, 1)
            pl.set("uMode", mode)
            pl.set("uUseNormalMap", int(self.use_normal_map))
            pl.set("uUseAO", int(self.use_ao))
            pl.set("uDirectX", int(self.directx))
            pl.set("uWire", int(self.show_wire))
            pl.set("uWireCol", (0.95, 0.6, 0.2) if self.channel == "lit" else (0.1, 0.1, 0.1))
            pl.set("uOffset", 0.0)
            names = ("tAlbedo", "tNormal", "tAO", "tRough", "tMetal", "tEmit", "tChannel")
            for i, n in enumerate(names):
                pl.set(n, i)
            for g in self.groups:
                keys = ("albedo", "normal", "ao", "roughness", "metalness", "emissive")
                for i, k in enumerate(keys):
                    GL.glActiveTexture(GL.GL_TEXTURE0 + i)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex(g["set"], k))
                GL.glActiveTexture(GL.GL_TEXTURE0 + 6)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex(g["set"], self.channel)
                                 if mode == 1 else self.defaults["_black"])
                if self.show_high:     # push the low slightly back so the high can win where coplanar
                    GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
                    GL.glPolygonOffset(1.0, 1.0)
                g["low"].draw()
                GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glActiveTexture(GL.GL_TEXTURE0)

        if self.show_high:
            self.p_solid.use()
            self.p_solid.set("uVP", VP)
            self.p_solid.set("uCam", eye)
            self.p_solid.set("uLightDir", light)
            self.p_solid.set("uColor", (0.72, 0.74, 0.78))
            self.p_solid.set("uAlpha", 1.0)
            for g in self.groups:
                for h in g["highs"]:
                    h.draw()

        if self.show_cage:
            self.p_line.use()
            self.p_line.set("uVP", VP)
            self.p_line.set("uColor", (1.0, 0.45, 0.15, 1.0))
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            for g in self.groups:
                self.p_line.set("uOffset", float(self.cage_offsets.get(g["name"], 0.05)))
                GL.glDepthFunc(GL.GL_LEQUAL)
                g["low"].draw_lines()
                # faint pass through geometry so the whole cage is readable
                GL.glDepthFunc(GL.GL_GREATER)
                self.p_line.set("uColor", (1.0, 0.45, 0.15, 0.22))
                g["low"].draw_lines()
                self.p_line.set("uColor", (1.0, 0.45, 0.15, 1.0))
            GL.glDepthFunc(GL.GL_LEQUAL)
            GL.glDisable(GL.GL_BLEND)
        GL.glBindVertexArray(0)

    # ------------------------------------------------------------- input
    def mousePressEvent(self, e):
        self._drag = (e.position(), e.buttons(), e.modifiers())

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseMoveEvent(self, e):
        if not self._drag:
            return
        p0, btn, mods = self._drag
        d = e.position() - p0
        self._drag = (e.position(), btn, mods)
        if btn & Qt.LeftButton and mods & Qt.ShiftModifier or btn & Qt.MiddleButton or \
                btn & Qt.RightButton and not mods & Qt.ShiftModifier:
            self._pan(d)
        elif btn & Qt.RightButton:                     # shift + right drag: rotate light
            self.light_yaw = (self.light_yaw + d.x() * 0.5) % 360
        elif btn & Qt.LeftButton:
            self.yaw = (self.yaw - d.x() * 0.4) % 360
            self.pitch = max(-89.0, min(89.0, self.pitch + d.y() * 0.4))
        self.update()

    def _pan(self, d: QPointF):
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        fwd = np.array([math.sin(y) * math.cos(p), -math.cos(y) * math.cos(p), math.sin(p)])
        right = np.cross(fwd, (0, 0, 1))
        right /= max(np.linalg.norm(right), 1e-6)
        up = np.cross(right, fwd)
        k = self.dist * 2 * math.tan(math.radians(20)) / max(self.height(), 1)
        self.target = (self.target + (-d.x() * right + d.y() * up) * k).astype(np.float32)

    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120.0
        self.dist = max(self.radius * 0.02, self.dist * (0.85 ** steps))
        self.update()

    def mouseDoubleClickEvent(self, e):
        self.frame()

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_F:
            self.frame()
        else:
            e.ignore()          # let the panel handle toggles


def _unique_edges(P):
    """Edges of an unindexed triangle list, welded by position (for cage/wire lines)."""
    q = np.round(P / max(float(np.abs(P).max()), 1e-9) * 1e6).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    tri = np.arange(len(P)).reshape(-1, 3)
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    ew = np.sort(inv[e], axis=1)
    _, first = np.unique(ew, axis=0, return_index=True)
    return e[first]


# ---------------------------------------------------------------------------
# Panel: toolbar + viewport
# ---------------------------------------------------------------------------
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox,  # noqa: E402
                               QToolButton, QLabel, QSlider, QStackedLayout)


class ViewerPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)
        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.channel = QComboBox()
        for key, label in CHANNELS:
            self.channel.addItem(label, key)
        self.channel.setToolTip("What to show on the low poly (keys 1-9 cycle through baked maps)")
        bar.addWidget(self.channel)
        self.cb = {}
        for key, label, tip, default in (
                ("normal", "Normal map", "Apply the baked normal map (N)", True),
                ("ao", "AO", "Multiply the baked AO into the lighting", True),
                ("high", "High", "Show the high poly (H)", False),
                ("cage", "Cage", "Show the ray cage at each group's cage offset (C)", False),
                ("wire", "Wire", "Wireframe overlay on the low poly (W)", False),
                ("low", "Low", "Show the low poly (L)", True)):
            c = QCheckBox(label)
            c.setToolTip(tip)
            c.setChecked(default)
            self.cb[key] = c
            bar.addWidget(c)
        bar.addSpacing(8)
        bar.addWidget(QLabel("Light"))
        self.light = QSlider(Qt.Horizontal)
        self.light.setRange(0, 359)
        self.light.setValue(40)
        self.light.setMaximumWidth(110)
        self.light.setToolTip("Rotate the light (or Shift + right-drag in the view)")
        bar.addWidget(self.light)
        bar.addWidget(QLabel("Exposure"))
        self.expo = QSlider(Qt.Horizontal)
        self.expo.setRange(-30, 30)
        self.expo.setValue(0)
        self.expo.setMaximumWidth(90)
        bar.addWidget(self.expo)
        bar.addStretch(1)
        fr = QToolButton(text="Frame")
        fr.setToolTip("Frame the model (F or double-click)")
        bar.addWidget(fr)
        v.addLayout(bar)

        self.view = Viewport()
        self.placeholder = QLabel()
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setObjectName("hint")
        self.stack = QStackedLayout()
        self.stack.addWidget(self.view)
        self.stack.addWidget(self.placeholder)
        holder = QWidget()
        holder.setLayout(self.stack)
        v.addWidget(holder, 1)
        self.info = QLabel("Left-drag orbit · right/middle-drag pan · wheel zoom · "
                           "Shift+right-drag rotate light · F frame")
        self.info.setObjectName("hint")
        v.addWidget(self.info)

        self.channel.currentIndexChanged.connect(self._sync)
        for c in self.cb.values():
            c.toggled.connect(self._sync)
        self.light.valueChanged.connect(self._sync)
        self.expo.valueChanged.connect(self._sync)
        fr.clicked.connect(self.view.frame)
        self.view.status.connect(self._status)
        self._sync()
        self._checked = False

    def showEvent(self, e):
        super().showEvent(e)
        if not self._checked:
            self._checked = True
            from PySide6.QtCore import QTimer
            QTimer.singleShot(2000, self._check_gl)

    def _check_gl(self):
        # if Qt could not create a GL context, initializeGL never runs: explain instead of a blank view
        if not self.view.ok:
            self._status("OpenGL unavailable: " + (self.view.error or "could not create an "
                                                   "OpenGL 3.3 context on this system"))

    def _status(self, msg):
        if msg.startswith("OpenGL unavailable"):
            self.placeholder.setText("The live 3D viewer needs OpenGL 3.3.\n" + msg +
                                     "\n\nBaking and the Cycles render preview still work.")
            self.stack.setCurrentIndex(1)
        elif not msg.startswith("OpenGL"):
            self.info.setText(msg + "   ·   Left-drag orbit · right-drag pan · wheel zoom · "
                                    "Shift+right-drag light · F frame")

    def _sync(self):
        v = self.view
        v.channel = self.channel.currentData()
        v.use_normal_map = self.cb["normal"].isChecked()
        v.use_ao = self.cb["ao"].isChecked()
        v.show_high = self.cb["high"].isChecked()
        v.show_cage = self.cb["cage"].isChecked()
        v.show_wire = self.cb["wire"].isChecked()
        v.show_low = self.cb["low"].isChecked()
        v.light_yaw = float(self.light.value())
        v.exposure = 2.0 ** (self.expo.value() / 10.0)
        v.update()

    def set_directx(self, on):
        self.view.directx = bool(on)
        self.view.update()

    def keyPressEvent(self, e):
        keys = {Qt.Key_N: "normal", Qt.Key_H: "high", Qt.Key_C: "cage", Qt.Key_W: "wire",
                Qt.Key_L: "low"}
        if e.key() in keys:
            c = self.cb[keys[e.key()]]
            c.setChecked(not c.isChecked())
        elif Qt.Key_0 <= e.key() <= Qt.Key_9:
            i = e.key() - Qt.Key_0
            if i < self.channel.count():
                self.channel.setCurrentIndex(i)
        else:
            super().keyPressEvent(e)
