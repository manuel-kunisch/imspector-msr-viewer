"""
msr_volume_view.py -- the OpenGL widget of the 3D view: GPU ray casting, interaction and overlays.

The ray caster of msr_volume.render_cpu as a GLSL fragment shader: one full-screen
triangle, every pixel marches its own ray through one 3D texture per channel.  After
the OpenCL kernel of the Fiji plugin 3Dscript (B. Schmid et al., "3Dscript: animating
3D/4D microscopy data using a natural-language-based syntax", Nature Methods
16:278-280, 2019; https://github.com/bene51/3Dscript): like 3Dscript the shader source
is generated per number of channels and rendering algorithm, with the per-channel
code unrolled.  Portions are derived from 3Dscript, Copyright (c) 2016, Benjamin
Schmid, under the BSD 2-Clause License; see THIRD_PARTY_NOTICES.md.

* GLRenderer: GL resources (programs, the 3D textures, the lookup-table texture) and
  the draw call, for whichever context is current.
* VolumeView: the QOpenGLWidget, with 3Dscript's mouse interaction, coarser passes
  while the view changes, and the overlays (bounding box, axes, scale bar, message).
* OffscreenRenderer: a GLRenderer with its own context on an offscreen surface (tests,
  exports).
* VolumeLoadThread: msr_volume.load_volume in the background (never touches GL).

Only PyQt5 and numpy are needed (no PyOpenGL): GL functions come from
QOpenGLContext.versionFunctions, textures are uploaded with QOpenGLTexture (PyQt5's
glTexImage3D with data crashes).  Desktop OpenGL 3.0 or newer is required: the shaders
are GLSL 3.30 core on OpenGL 3.3+ (macOS core profiles) and GLSL 1.30 on 3.0-3.2, which
is what Qt's software renderer (Mesa llvmpipe, MSR_VIEWER_SOFTWARE_GL=1) provides.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, sip
from PyQt5.QtCore import Qt

from msr_volume import (ALGORITHMS, DEFAULT_BUDGET, DEFAULT_MAX_DIM, MAX_CHANNELS, PRESETS, ROT_SPEED, SATURATED,
                        TF_EPS, WHEEL_STEP, Camera, ChannelSettings, RenderSettings, Volume, axes_tripod, box_edges,
                        load_volume, ray_setup, rendered_channels)
from msr_widgets import paint_scalebar

# GL enums (PyQt5 has no GL constants module)
GL_TRIANGLES, GL_FLOAT = 0x0004, 0x1406
GL_DEPTH_TEST, GL_BLEND, GL_SCISSOR_TEST, GL_CULL_FACE = 0x0B71, 0x0BE2, 0x0C11, 0x0B44
GL_STENCIL_TEST, GL_DITHER, GL_FRAMEBUFFER_SRGB = 0x0B90, 0x0BD0, 0x8DB9
GL_TEXTURE_2D, GL_TEXTURE0 = 0x0DE1, 0x84C0
GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR = 0x2801, 0x2800, 0x2601
GL_RENDERER, GL_VERSION = 0x1F01, 0x1F02
GL_MAX_3D_TEXTURE_SIZE = 0x8073
GL_RGBA8 = 0x8058
GL_OUT_OF_MEMORY = 0x0505
GL_UNPACK_ALIGNMENT = 0x0CF5

STRIP_ROWS = 256           # rows per draw call of full-quality passes (keeps each call short: GPU watchdog)
INTERACTIVE_STEP = 2.5     # step_scale of the passes while the view changes
SLOW_FRAME_MS = 40.0       # full-quality frame time above which interactive passes use half resolution
FAST_FRAME_MS = 12.0       # ... below which interactive passes are rendered at full quality anyway
SETTLE_MS = 200            # full-quality pass this long after the last change
AXIS_COLORS = {"x": (235, 85, 85), "y": (95, 205, 95), "z": (95, 145, 255)}
NO_VOLUME_BG = (24 / 255, 24 / 255, 26 / 255)
NO_CONTEXT = "The 3D view could not create an OpenGL context. It needs OpenGL 3.3 (3.0 at least)."


def _pip_qt() -> bool:
    """True if Qt comes with PyQt5 itself (pip wheel, or the packaged app built from one)."""
    if getattr(sys, "frozen", False):
        return True
    try:
        import PyQt5
        qt_bin = os.path.normcase(os.path.abspath(QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.BinariesPath)))
        return qt_bin.startswith(os.path.normcase(os.path.dirname(os.path.abspath(PyQt5.__file__))) + os.sep)
    except Exception:
        return False


def gl_advice() -> str:
    """What to do about a missing or too old OpenGL, for the error messages.

    Qt's software renderer (AA_UseSoftwareOpenGL, MSR_VIEWER_SOFTWARE_GL=1) exists only in
    Windows builds of Qt that ship opengl32sw.dll and honour the attribute: the packaged
    Windows app and PyQt5 from pip, not conda-forge's Qt, and not macOS.
    """
    if sys.platform != "win32":
        return "Update the graphics driver (or the operating system)."
    if os.environ.get("MSR_VIEWER_SOFTWARE_GL", "").strip() not in ("", "0"):
        if _pip_qt():
            return "Qt's software renderer (MSR_VIEWER_SOFTWARE_GL) did not work either. Update the graphics driver."
        return ("Update the graphics driver. MSR_VIEWER_SOFTWARE_GL has no effect with this Qt build (e.g. "
                "conda-forge's); Qt's software renderer is part of the packaged Windows app and of PyQt5 from pip.")
    if _pip_qt():
        return ("Update the graphics driver, or start the viewer with the environment variable "
                "MSR_VIEWER_SOFTWARE_GL=1 to use Qt's software renderer (much slower).")
    return ("Update the graphics driver, or use the packaged Windows app or PyQt5 from pip, where the environment "
            "variable MSR_VIEWER_SOFTWARE_GL=1 selects Qt's software renderer (much slower; conda-forge's Qt "
            "has none).")


class _NoContext(RuntimeError):
    """No OpenGL context could be created."""


# -----------------------------------------------------------------------------
# shaders
# -----------------------------------------------------------------------------

# "{version}" in the sources: GLSL_330 on OpenGL 3.3+, GLSL_130 on 3.0-3.2 (GLRenderer.initialize)
GLSL_330, GLSL_130 = "#version 330 core", "#version 130"
# one triangle that covers the viewport (a vertex buffer, not gl_VertexID: Mesa's compatibility
# profile draws nothing without an enabled attribute array)
_TRIANGLE = np.array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0], np.float32)

_VERTEX = """{version}
in vec2 a_pos;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

_COPY = """{version}
uniform sampler2D u_tex;
uniform vec2 u_viewport;
out vec4 fragColor;
void main() {
    fragColor = vec4(texture(u_tex, gl_FragCoord.xy / u_viewport).rgb, 1.0);
}
"""

_FRAG_HEAD = """{version}
// ray caster of msr_volume (after 3Dscript's OpenCL kernel): {n} channel(s), {algorithm}
out vec4 fragColor;
uniform mat4 u_M;          // (px, py, s, 1) -> texture coordinates (msr_volume.ray_setup)
uniform vec2 u_size;       // output size in pixels
uniform vec2 u_origin;     // output pixel of the viewport origin (strips and tiles)
uniform float u_step;      // sample distance in s units
uniform float u_alphacorr; // opacity correction exponent
uniform int u_maxsteps;
uniform vec3 u_bg;
uniform vec3 u_crop0;
uniform vec3 u_crop1;
uniform float u_dalpha;    // 1: 3Dscript's A*C + (1-A)*bg, 0: C + (1-A)*bg
uniform sampler2D u_lut;   // 256 x {maxch} RGBA, one row per rendered channel
"""

_FRAG_FUNCS = """
vec3 lut(float r, float row) {
    return textureLod(u_lut, vec2((r * 255.0 + 0.5) / 256.0, (row + 0.5) / {maxch}.0), 0.0).rgb;
}

bool slab(vec3 u0, vec3 du, out float s0, out float s1) {
    s0 = -1e30;
    s1 = 1e30;
    for (int i = 0; i < 3; ++i) {
        if (abs(du[i]) < 1e-12) {
            if (u0[i] < u_crop0[i] || u0[i] > u_crop1[i]) return false;
        } else {
            float t1 = (u_crop0[i] - u0[i]) / du[i];
            float t2 = (u_crop1[i] - u0[i]) / du[i];
            s0 = max(s0, min(t1, t2));
            s1 = min(s1, max(t1, t2));
        }
    }
    return s1 > s0;
}
"""


def fragment_source(n: int, algorithm: str, version: str = GLSL_330) -> str:
    """GLSL of the ray caster for *n* rendered channels and *algorithm* (per-channel code unrolled)."""
    mip, combined = algorithm == "mip", algorithm == "combined"
    ch = range(n)
    src = [_FRAG_HEAD.replace("{version}", version).replace("{n}", str(n)).replace("{algorithm}", algorithm)
           .replace("{maxch}", str(MAX_CHANNELS))]
    for c in ch:
        src.append(f"uniform sampler3D u_vol{c};\n"
                   f"uniform float u_cmin{c}, u_cscale{c}, u_cgamma{c}, u_amin{c}, u_ascale{c}, u_agamma{c}, "
                   f"u_weight{c};")
    src.append(_FRAG_FUNCS.replace("{maxch}", str(MAX_CHANNELS)))
    src.append("void main() {")
    if n == 0:
        src.append("    fragColor = vec4(u_bg, 1.0);\n}\n")
        return "\n".join(src)
    src.append("""    vec2 pix = vec2(gl_FragCoord.x + u_origin.x, u_size.y - (gl_FragCoord.y + u_origin.y));
    vec3 u0 = (u_M * vec4(pix, 0.0, 1.0)).xyz;
    vec3 du = u_M[2].xyz;
    float s0, s1;
    if (!slab(u0, du, s0, s1)) {
        fragColor = vec4(u_bg, 1.0);
        return;
    }""")
    for c in ch:
        src.append(f"    float best{c} = -1.0, vbest{c} = 0.0;" if mip else
                   f"    vec3 color{c} = vec3(0.0);\n    float alpha{c} = 0.0;")
    src.append("""    for (int k = 0; k < u_maxsteps; ++k) {
        float s = s0 + (float(k) + 0.5) * u_step;
        if (s >= s1) break;
        vec3 p = u0 + s * du;""")
    for c in ch:
        if mip:   # compare the clamped ramp (t^gamma is monotonic); the first maximum wins
            src.append(f"""        {{
            float v = textureLod(u_vol{c}, p, 0.0).r;
            float t = clamp((v - u_cmin{c}) * u_cscale{c}, 0.0, 1.0);
            if (t > best{c}) {{ best{c} = t; vbest{c} = v; }}
        }}""")
        else:
            a = " + ".join(f"alpha{o}" for o in ch) if combined else f"alpha{c}"
            src.append(f"""        if (alpha{c} < {SATURATED}) {{
            float v = textureLod(u_vol{c}, p, 0.0).r;
            float rc = pow(clamp((v - u_cmin{c}) * u_cscale{c}, 0.0, 1.0), u_cgamma{c});
            float ra = pow(clamp((v - u_amin{c}) * u_ascale{c}, 0.0, 1.0), u_agamma{c});
            ra = 1.0 - pow(1.0 - ra, u_alphacorr);
            float tmp = u_weight{c} * max(1.0 - ({a}), 0.0) * ra;
            color{c} += tmp * lut(rc, {c}.0);
            alpha{c} += tmp;
        }}""")
    if mip:
        stop = " && ".join(f"best{c} >= 1.0" for c in ch)
    elif combined:
        stop = " + ".join(f"alpha{c}" for c in ch) + f" >= {SATURATED}"
    else:
        stop = " && ".join(f"alpha{c} >= {SATURATED}" for c in ch)
    src.append(f"        if ({stop}) break;\n    }}")
    if mip:
        src.append("    vec3 outc = vec3(0.0);\n    float A = 0.0;")
        for c in ch:
            src.append(f"""    if (best{c} >= 0.0) {{
        outc += u_weight{c} * lut(pow(best{c}, u_cgamma{c}), {c}.0);
        A += u_weight{c} * pow(clamp((vbest{c} - u_amin{c}) * u_ascale{c}, 0.0, 1.0), u_agamma{c});
    }}""")
        src.append("""    A = clamp(A, 0.0, 1.0);
    fragColor = vec4(clamp(outc + (1.0 - A) * u_bg, 0.0, 1.0), 1.0);
}
""")
    else:
        src.append("    vec3 outc = " + " + ".join(f"u_weight{c} * color{c}" for c in ch) + ";")
        src.append("    float A = clamp(" + " + ".join(f"u_weight{c} * alpha{c}" for c in ch) + ", 0.0, 1.0);")
        src.append("""    vec3 pixc = mix(outc, A * outc, u_dalpha) + (1.0 - A) * u_bg;
    fragColor = vec4(clamp(pixc, 0.0, 1.0), 1.0);
}
""")
    return "\n".join(src)


def _uniform_names(n: int) -> list[str]:
    names = ["u_M", "u_size", "u_origin", "u_step", "u_alphacorr", "u_maxsteps", "u_bg", "u_crop0", "u_crop1",
             "u_dalpha", "u_lut", "u_tex", "u_viewport"]
    for c in range(n):
        names += [f"u_{k}{c}" for k in ("vol", "cmin", "cscale", "cgamma", "amin", "ascale", "agamma", "weight")]
    return names


def _gl_functions(ctx: QtGui.QOpenGLContext):
    """The context's GL functions: 4.1 core (core and compatibility contexts), else 2.0; None if neither."""
    for major, minor, core in ((4, 1, True), (2, 0, False)):
        profile = QtGui.QOpenGLVersionProfile()
        profile.setVersion(major, minor)
        if core:
            profile.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        try:
            gl = ctx.versionFunctions(profile)
        except Exception:  # module missing from a frozen build, unsupported profile
            gl = None
        if gl is not None and gl.initializeOpenGLFunctions():
            return gl
    return None


def _gl_string(gl, name: int) -> str:
    try:
        s = gl.glGetString(name)
    except Exception:
        return ""
    return s.decode("latin-1", "replace") if isinstance(s, bytes) else str(s or "")


_TEX_FORMATS = {
    np.dtype(np.uint8): (QtGui.QOpenGLTexture.R8_UNorm, QtGui.QOpenGLTexture.UInt8),
    np.dtype(np.uint16): (QtGui.QOpenGLTexture.R16_UNorm, QtGui.QOpenGLTexture.UInt16),
    np.dtype(np.float32): (QtGui.QOpenGLTexture.R32F, QtGui.QOpenGLTexture.Float32),
}


def _set_data(gl, tex: QtGui.QOpenGLTexture, pixel_format, pixel_type, data: np.ndarray) -> None:
    """tex.setData from a C-contiguous array whose rows are tightly packed.

    The row alignment is set by hand: QOpenGLPixelTransferOptions never reach GL through
    PyQt5 5.15, and GL's default of 4 bytes would shear rows of 1- or 2-byte voxels whose
    length is not a multiple of 4 bytes and read past the end of the array.
    """
    gl.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    try:
        tex.setData(pixel_format, pixel_type, sip.voidptr(data.ctypes.data, data.nbytes))
    finally:
        gl.glPixelStorei(GL_UNPACK_ALIGNMENT, 4)   # GL's default, which Qt's own uploads expect


# -----------------------------------------------------------------------------
# renderer
# -----------------------------------------------------------------------------

class GLRenderer:
    """GL resources of the ray caster and its draw call; works in whichever context is current.

    Every method except the constructor needs the context of initialize() to be current.
    """

    def __init__(self):
        self.gl = None
        self.info = ""                    # renderer and GL version, after initialize()
        self.max_3d = DEFAULT_MAX_DIM     # GL_MAX_3D_TEXTURE_SIZE, after initialize()
        self._programs: dict[tuple[int, str], tuple[QtGui.QOpenGLShaderProgram, dict[str, int]]] = {}
        self._failed: dict[tuple[int, str], str] = {}   # programs the driver could not link: the error text
        self._copy = None
        self._vao = None
        self._vbo = None
        self._glsl = GLSL_330
        self._textures: list[QtGui.QOpenGLTexture] = []
        self._uploaded: list[np.ndarray] = []
        self._lut = None
        self._lut_data = None

    # -- set-up ---------------------------------------------------------------------

    def initialize(self, ctx: QtGui.QOpenGLContext) -> str | None:
        """Get the GL functions and create the shared resources; the error text if GL cannot be used."""
        fmt = ctx.format()
        version = (fmt.majorVersion(), fmt.minorVersion())
        if ctx.isOpenGLES() or version < (3, 0):
            have = f"OpenGL {'ES ' if ctx.isOpenGLES() else ''}{version[0]}.{version[1]}"
            return f"The 3D view needs OpenGL 3.0 or newer (this computer provides {have}). {gl_advice()}"
        gl = _gl_functions(ctx)
        if gl is None:
            return "The 3D view could not load the OpenGL functions of this graphics driver."
        self.gl = gl
        self._glsl = GLSL_330 if version >= (3, 3) else GLSL_130
        self.info = f"{_gl_string(gl, GL_RENDERER) or 'unknown renderer'}, OpenGL {_gl_string(gl, GL_VERSION)}"
        try:
            v = gl.glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE)
            v = int(v[0] if isinstance(v, (tuple, list)) else v)
            self.max_3d = v if v > 0 else DEFAULT_MAX_DIM
        except Exception:
            self.max_3d = DEFAULT_MAX_DIM
        self._vao = QtGui.QOpenGLVertexArrayObject()
        if not self._vao.create():
            self.release()
            return "The 3D view could not create an OpenGL vertex array object."
        try:
            self._copy = self._link(_COPY.replace("{version}", self._glsl), 0)
            self._vao.bind()         # the triangle's vertex buffer is part of the VAO state
            self._vbo = QtGui.QOpenGLBuffer(QtGui.QOpenGLBuffer.VertexBuffer)
            self._vbo.create()
            self._vbo.bind()
            self._vbo.allocate(sip.voidptr(_TRIANGLE.ctypes.data, _TRIANGLE.nbytes), _TRIANGLE.nbytes)
            self._copy[0].enableAttributeArray(0)
            self._copy[0].setAttributeBuffer(0, GL_FLOAT, 0, 2)
            self._vao.release()
            self._vbo.release()
            self._program(1, "independent")   # compile one ray caster now: shader errors show up at once
        except RuntimeError as exc:
            self.release()
            return str(exc)
        self._lut = QtGui.QOpenGLTexture(QtGui.QOpenGLTexture.Target2D)
        self._lut.setFormat(QtGui.QOpenGLTexture.RGBA8_UNorm)
        self._lut.setSize(256, MAX_CHANNELS)
        self._lut.setMipLevels(1)
        self._lut.allocateStorage(QtGui.QOpenGLTexture.RGBA, QtGui.QOpenGLTexture.UInt8)
        self._lut.setMinMagFilters(QtGui.QOpenGLTexture.Linear, QtGui.QOpenGLTexture.Linear)
        self._lut.setWrapMode(QtGui.QOpenGLTexture.ClampToEdge)
        self._lut_data = None
        return None

    def max_texture_size(self) -> int:
        """Largest 3D texture side of the driver (DEFAULT_MAX_DIM before initialize)."""
        return self.max_3d

    def _link(self, fragment: str, n: int) -> tuple[QtGui.QOpenGLShaderProgram, dict[str, int]]:
        prog = QtGui.QOpenGLShaderProgram()
        prog.bindAttributeLocation("a_pos", 0)
        if not (prog.addShaderFromSourceCode(QtGui.QOpenGLShader.Vertex, _VERTEX.replace("{version}", self._glsl))
                and prog.addShaderFromSourceCode(QtGui.QOpenGLShader.Fragment, fragment) and prog.link()):
            log = prog.log().strip()
            sip.delete(prog)     # GL object: delete while the context is current
            raise RuntimeError(f"The 3D view could not compile its shader:\n{log}")
        return prog, {name: prog.uniformLocation(name) for name in _uniform_names(n)}

    def _program(self, n: int, algorithm: str) -> tuple[QtGui.QOpenGLShaderProgram, dict[str, int]]:
        """The ray caster for *n* channels; RuntimeError if the driver cannot link it.

        The first request for *n* channels links every algorithm, so a driver problem shows
        up as soon as that many channels are rendered; a failed link is not retried.
        """
        key = (n, algorithm if n else "")
        if key not in self._programs and key not in self._failed:
            for k in ([(n, a) for a in ALGORITHMS] if n else [(0, "")]):
                if k not in self._programs and k not in self._failed:
                    try:
                        self._programs[k] = self._link(fragment_source(n, k[1], self._glsl), n)
                    except RuntimeError as exc:
                        self._failed[k] = str(exc)
        if key in self._failed:
            raise RuntimeError(self._failed[key])
        return self._programs[key]

    # -- data -----------------------------------------------------------------------

    def upload(self, volume: Volume | None) -> None:
        """Upload the channels of *volume* as 3D textures (nothing to do if they are already up)."""
        arrays = [c.data for c in volume.channels] if volume is not None else []
        if len(arrays) == len(self._uploaded) and all(a is b for a, b in zip(arrays, self._uploaded)):
            return
        self.release_volume()
        try:
            for a in arrays:
                self._textures.append(self._texture3d(a))
        except Exception:
            self.release_volume()
            raise
        self._uploaded = arrays

    def _texture3d(self, data: np.ndarray) -> QtGui.QOpenGLTexture:
        gl = self.gl
        if data.dtype not in _TEX_FORMATS:
            data = data.astype(np.float32)
        data = np.ascontiguousarray(data)
        fmt, ptype = _TEX_FORMATS[data.dtype]
        d, h, w = data.shape
        for _ in range(16):          # clear stale errors so the check below is about this upload
            if not gl.glGetError():
                break
        tex = QtGui.QOpenGLTexture(QtGui.QOpenGLTexture.Target3D)
        tex.setFormat(fmt)
        tex.setSize(w, h, d)
        tex.setMipLevels(1)
        tex.allocateStorage(QtGui.QOpenGLTexture.Red, ptype)
        _set_data(gl, tex, QtGui.QOpenGLTexture.Red, ptype, data)
        tex.setMinMagFilters(QtGui.QOpenGLTexture.Linear, QtGui.QOpenGLTexture.Linear)
        tex.setWrapMode(QtGui.QOpenGLTexture.ClampToEdge)
        err = gl.glGetError()
        if err or not tex.isStorageAllocated():
            tex.destroy()
            why = " (out of graphics memory)" if err == GL_OUT_OF_MEMORY else ""
            raise RuntimeError(f"OpenGL error 0x{err:04x} while uploading a {w} x {h} x {d} volume{why}")
        return tex

    def _set_luts(self, tables: list[np.ndarray]) -> None:
        data = np.zeros((MAX_CHANNELS, 256, 4), np.uint8)
        data[..., 3] = 255
        for i, t in enumerate(tables[:MAX_CHANNELS]):
            data[i, :, :3] = t
        if self._lut_data is not None and np.array_equal(data, self._lut_data):
            return
        _set_data(self.gl, self._lut, QtGui.QOpenGLTexture.RGBA, QtGui.QOpenGLTexture.UInt8, data)
        self._lut_data = data

    # -- drawing --------------------------------------------------------------------

    def _reset_state(self) -> None:
        """The state the ray caster needs (QPainter changes it between frames)."""
        gl = self.gl
        for cap in (GL_DEPTH_TEST, GL_BLEND, GL_SCISSOR_TEST, GL_CULL_FACE, GL_STENCIL_TEST, GL_DITHER,
                    GL_FRAMEBUFFER_SRGB):
            gl.glDisable(cap)
        gl.glColorMask(True, True, True, True)

    def draw(self, channels: list[ChannelSettings], settings: RenderSettings, camera: Camera,
             volume: Volume | None, w: int, h: int, step_scale: float = 1.0,
             rows: tuple[int, int] | None = None) -> None:
        """Ray cast an image of w x h pixels into the bound framebuffer.

        *rows* = (first GL row, count) draws only that strip of the image (render_image).
        The volume must have been uploaded; without it, or without a rendered channel, the
        output is the background colour.
        """
        gl = self.gl
        w, h = max(1, int(w)), max(1, int(h))
        y0, n_rows = rows if rows is not None else (0, h)
        self._reset_state()
        gl.glViewport(0, int(y0), w, int(n_rows))
        active = rendered_channels(channels, volume) if self._uploaded else []
        active = [i for i in active if i < len(self._textures)]
        algorithm = settings.algorithm if settings.algorithm in ALGORITHMS else "independent"
        prog, loc = self._program(len(active), algorithm)
        if active:   # LUT data first: setData may bind the texture to the active unit
            self._set_luts([channels[i].lut_table() for i in active])
        prog.bind()
        try:
            bg = [float(min(max(c, 0.0), 1.0)) for c in settings.background[:3]]
            if volume is None:
                bg = list(NO_VOLUME_BG)
            gl.glUniform3f(loc["u_bg"], *bg)
            if active:
                rs = ray_setup(camera, volume, w, h, settings.sampling, step_scale)
                prog.setUniformValue(loc["u_M"], QtGui.QMatrix4x4(*[float(v) for v in rs.tex_from_pixel.ravel()]))
                crop = [float(c) for c in settings.crop]
                gl.glUniform2f(loc["u_size"], float(w), float(h))
                gl.glUniform2f(loc["u_origin"], 0.0, 0.0)
                gl.glUniform1f(loc["u_step"], float(rs.step))
                gl.glUniform1f(loc["u_alphacorr"], float(rs.alphacorr))
                gl.glUniform1i(loc["u_maxsteps"], int(rs.max_steps))
                gl.glUniform3f(loc["u_crop0"], crop[0], crop[2], crop[4])
                gl.glUniform3f(loc["u_crop1"], crop[1], crop[3], crop[5])
                gl.glUniform1f(loc["u_dalpha"], 1.0 if settings.double_alpha else 0.0)
                for k, i in enumerate(active):
                    cs, vc = channels[i], volume.channels[i]
                    vs = vc.value_scale     # the textures hold raw / vs: the windows go to texture units
                    self._textures[i].bind(k)
                    gl.glUniform1i(loc[f"u_vol{k}"], k)
                    gl.glUniform1f(loc[f"u_cmin{k}"], float(cs.cmin / vs))
                    gl.glUniform1f(loc[f"u_cscale{k}"], float(vs / max(cs.cmax - cs.cmin, TF_EPS)))
                    gl.glUniform1f(loc[f"u_cgamma{k}"], float(cs.cgamma))
                    gl.glUniform1f(loc[f"u_amin{k}"], float(cs.amin / vs))
                    gl.glUniform1f(loc[f"u_ascale{k}"], float(vs / max(cs.amax - cs.amin, TF_EPS)))
                    gl.glUniform1f(loc[f"u_agamma{k}"], float(cs.agamma))
                    gl.glUniform1f(loc[f"u_weight{k}"], float(cs.weight))
                self._lut.bind(len(active))
                gl.glUniform1i(loc["u_lut"], len(active))
            self._vao.bind()
            gl.glDrawArrays(GL_TRIANGLES, 0, 3)
            self._vao.release()
        finally:
            prog.release()
            gl.glActiveTexture(GL_TEXTURE0)

    def blit(self, texture_id: int, w: int, h: int) -> None:
        """Draw a 2D texture over the whole w x h viewport, linearly interpolated (half-resolution passes)."""
        gl = self.gl
        prog, loc = self._copy
        self._reset_state()
        gl.glViewport(0, 0, int(w), int(h))
        prog.bind()
        gl.glActiveTexture(GL_TEXTURE0)
        gl.glBindTexture(GL_TEXTURE_2D, int(texture_id))
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        gl.glUniform1i(loc["u_tex"], 0)
        gl.glUniform2f(loc["u_viewport"], float(w), float(h))
        self._vao.bind()
        gl.glDrawArrays(GL_TRIANGLES, 0, 3)
        self._vao.release()
        prog.release()
        gl.glBindTexture(GL_TEXTURE_2D, 0)

    def render_to_image(self, channels: list[ChannelSettings], settings: RenderSettings, camera: Camera,
                        volume: Volume | None, w: int, h: int, step_scale: float = 1.0) -> QtGui.QImage:
        """Ray cast into a framebuffer object of w x h, in strips of STRIP_ROWS rows, and read it back (RGB32)."""
        w, h = max(1, int(w)), max(1, int(h))
        fmt = QtGui.QOpenGLFramebufferObjectFormat()
        fmt.setAttachment(QtGui.QOpenGLFramebufferObject.NoAttachment)
        fmt.setInternalTextureFormat(GL_RGBA8)
        fbo = QtGui.QOpenGLFramebufferObject(w, h, fmt)
        try:
            if not fbo.isValid() or not fbo.bind():
                raise RuntimeError(f"OpenGL could not create a {w} x {h} image")
            for y0 in range(0, h, STRIP_ROWS):
                self.draw(channels, settings, camera, volume, w, h, step_scale, rows=(y0, min(STRIP_ROWS, h - y0)))
                self.gl.glFinish()
            img = fbo.toImage()
            fbo.release()
        finally:
            sip.delete(fbo)      # GL object: delete while the context is current
        return img.convertToFormat(QtGui.QImage.Format_RGB32)

    # -- clean-up -------------------------------------------------------------------

    def release_volume(self) -> None:
        """Delete the volume textures."""
        for t in self._textures:
            t.destroy()
        self._textures, self._uploaded = [], []

    def release(self) -> None:
        """Delete every GL resource (the context must be current)."""
        if self.gl is None:
            return
        self.release_volume()
        if self._lut is not None:
            self._lut.destroy()
        if self._vao is not None:
            self._vao.destroy()
        if self._vbo is not None:
            self._vbo.destroy()
        for prog, _ in list(self._programs.values()) + ([self._copy] if self._copy else []):
            sip.delete(prog)
        self._programs, self._copy, self._vao, self._vbo, self._lut, self._lut_data = {}, None, None, None, None, None
        self._failed = {}
        self.gl = None


# -----------------------------------------------------------------------------
# overlays
# -----------------------------------------------------------------------------

def _qcolor(rgb) -> QtGui.QColor:
    return QtGui.QColor.fromRgbF(*[min(max(float(c), 0.0), 1.0) for c in rgb[:3]])


def paint_overlays(p: QtGui.QPainter, camera: Camera, volume: Volume | None, settings: RenderSettings,
                   w: float, h: float, message: str | None = None) -> None:
    """Bounding box, axes tripod, scale bar and a centred message on a w x h area (logical pixels)."""
    p.save()
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    if volume is not None and volume.channels:
        if settings.show_box:
            p.setPen(QtGui.QPen(_qcolor(settings.box_color), 1.0))
            for (x0, y0), (x1, y1) in box_edges(camera, volume, w, h):
                p.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))
        if settings.show_axes:
            _paint_tripod(p, camera, w, h)
        if settings.show_scalebar:
            paint_scalebar(p, w, h, camera.um_per_px(volume.extent, w, h), volume.unit)
    if message:
        _paint_message(p, message, w, h)
    p.restore()


def _paint_tripod(p: QtGui.QPainter, camera: Camera, w: float, h: float, length: float = 40.0) -> None:
    """The model axes x (red), y (green), z (blue) from a point at the lower right."""
    cx, cy = w - 24.0 - length, h - 24.0 - length
    font = QtGui.QFont(p.font())
    font.setPixelSize(11)
    font.setBold(True)
    p.setFont(font)
    for name, (dx, dy) in axes_tripod(camera, length):
        color = QtGui.QColor(*AXIS_COLORS[name])
        p.setPen(QtGui.QPen(color, 2.0, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(color)
        size = math.hypot(dx, dy)
        if size < 2.0:       # pointing at or away from the viewer
            p.drawEllipse(QtCore.QPointF(cx, cy), 2.5, 2.5)
            ux, uy = 0.7, -0.7
        else:
            p.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(cx + dx, cy + dy))
            ux, uy = dx / size, dy / size
        tx, ty = cx + dx + 9.0 * ux, cy + dy + 9.0 * uy
        p.drawText(QtCore.QRectF(tx - 8.0, ty - 8.0, 16.0, 16.0), Qt.AlignCenter, name)
    p.setBrush(Qt.NoBrush)


def _paint_message(p: QtGui.QPainter, text: str, w: float, h: float) -> None:
    font = QtGui.QFont(p.font())
    font.setPixelSize(13)
    p.setFont(font)
    fm = QtGui.QFontMetrics(font)
    avail = QtCore.QRect(0, 0, int(max(w - 64.0, 60.0)), 100000)
    br = fm.boundingRect(avail, int(Qt.AlignCenter | Qt.TextWordWrap), text)
    box = QtCore.QRectF((w - br.width()) / 2.0 - 14.0, (h - br.height()) / 2.0 - 10.0,
                        br.width() + 28.0, br.height() + 20.0)
    p.setPen(Qt.NoPen)
    p.setBrush(QtGui.QColor(0, 0, 0, 170))
    p.drawRoundedRect(box, 6.0, 6.0)
    p.setPen(QtGui.QColor(225, 225, 225))
    p.drawText(box, int(Qt.AlignCenter | Qt.TextWordWrap), text)


def _overlay_image(img: QtGui.QImage, camera: Camera, volume: Volume | None, settings: RenderSettings,
                   logical_w: float, logical_h: float) -> QtGui.QImage:
    """The overlays of a logical_w x logical_h view painted on *img*, scaled to its size."""
    scale = max(min(img.width() / max(logical_w, 1.0), img.height() / max(logical_h, 1.0)), 1e-3)
    p = QtGui.QPainter(img)
    p.scale(scale, scale)
    paint_overlays(p, camera, volume, settings, img.width() / scale, img.height() / scale)
    p.end()
    return img


# -----------------------------------------------------------------------------
# offscreen rendering
# -----------------------------------------------------------------------------

class OffscreenRenderer:
    """A GLRenderer with its own context on an offscreen surface (tests and exports; GUI thread only)."""

    def __init__(self, fmt: QtGui.QSurfaceFormat | None = None):
        fmt = fmt or QtGui.QSurfaceFormat.defaultFormat()
        self.surface = QtGui.QOffscreenSurface()
        self.surface.setFormat(fmt)
        self.surface.create()
        self.context = QtGui.QOpenGLContext()
        self.context.setFormat(fmt)
        if not self.context.create() or not self.context.makeCurrent(self.surface):
            raise _NoContext(f"{NO_CONTEXT} {gl_advice()}")
        self.renderer = GLRenderer()
        try:
            err = self.renderer.initialize(self.context)
        finally:
            self.context.doneCurrent()
        if err:
            raise RuntimeError(err)

    def render(self, volume: Volume | None, channels: list[ChannelSettings], settings: RenderSettings,
               camera: Camera, w: int, h: int, step_scale: float = 1.0, overlays: bool = False) -> QtGui.QImage:
        """The w x h image (RGB32), uploading the volume first if needed; overlays as if w x h were logical."""
        if not self.context.makeCurrent(self.surface):
            raise RuntimeError("could not make the OpenGL context current")
        try:
            self.renderer.upload(volume)
            img = self.renderer.render_to_image(channels, settings, camera, volume, w, h, step_scale)
        finally:
            self.context.doneCurrent()
        if overlays:
            img = _overlay_image(img, camera, volume, settings, w, h)
        return img

    def release(self) -> None:
        """Delete the GL resources and the context."""
        if self.context is None:
            return
        if self.context.makeCurrent(self.surface):
            self.renderer.release()
            self.context.doneCurrent()
        self.context = None
        self.surface.destroy()


def opengl_problem() -> str | None:
    """Why the 3D view cannot run on this computer, or None (GUI thread, QApplication needed).

    Sets up the ray caster in a context of the default format on an offscreen surface
    (version, GL functions, shader), so a missing or too old OpenGL is found before a
    QOpenGLWidget switches its whole window to OpenGL composition.  Creating the context
    is tried twice (a first attempt was seen to fail once with Qt's software renderer).
    """
    for attempt in (1, 2):
        try:
            OffscreenRenderer().release()
        except _NoContext as exc:
            if attempt == 1:
                continue
            return str(exc)
        except RuntimeError as exc:
            return str(exc)
        except Exception as exc:  # anything else from the driver or the bindings
            return f"The 3D view cannot use OpenGL on this computer: {type(exc).__name__}: {exc}"
        return None
    return f"{NO_CONTEXT} {gl_advice()}"


# -----------------------------------------------------------------------------
# the widget
# -----------------------------------------------------------------------------

class VolumeView(QtWidgets.QOpenGLWidget):
    """Interactive 3D view of a Volume: drag rotates, Shift- or middle-drag moves, the wheel zooms.

    The panel and the viewer edit channels / settings / camera in place and call
    settings_changed().  While the view changes, coarser passes are rendered (larger
    ray steps, half resolution when full frames are slow); SETTLE_MS after the last
    change a full-quality pass follows.
    """

    cameraChanged = QtCore.pyqtSignal()        # after a drag release, wheel, key, preset or set_volume
    glReady = QtCore.pyqtSignal(bool, str)     # GL initialised: ok, renderer description or error text

    _KEYS = {Qt.Key_R, Qt.Key_Home, Qt.Key_F, Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down,
             Qt.Key_Plus, Qt.Key_Minus, Qt.Key_Equal}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.channels: list[ChannelSettings] = []
        self.settings = RenderSettings()
        self.camera = Camera()
        self.volume: Volume | None = None
        self.renderer = GLRenderer()
        self.last_frame_ms = 0.0      # GPU time of the last frame (glFinish), any quality
        self.last_pass = ""           # "full", "interactive" (larger steps) or "half" (and half resolution)
        self.upload_ms = 0.0          # time of the last texture upload
        self._gl_error: str | None = None
        self._info = ""
        self._message: str | None = None
        self._upload_error: str | None = None
        self._draw_error: str | None = None   # shown instead of the volume until the next settings / volume change
        self._full_ms = 0.0           # time of the last full-quality frame
        self._half_fbo: QtGui.QOpenGLFramebufferObject | None = None
        self._press: QtCore.QPointF | None = None
        self._press_camera: Camera | None = None
        self._mode = "rotate"
        self._auto_fit = True         # re-fit on resize until the user moves the view
        self._fit_state: tuple | None = None   # zoom and pan of the last automatic fit
        self._settle = QtCore.QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(SETTLE_MS)
        self._settle.timeout.connect(self.update)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(False)
        self.setMinimumSize(160, 120)

    # -- API --------------------------------------------------------------------------

    def set_volume(self, volume: Volume | None, channels: list[ChannelSettings], keep_view: bool = False) -> None:
        """Show *volume* with *channels* (one per volume channel); the 3D preset unless the view is kept."""
        old = self.volume
        self.volume = volume
        self.channels = list(channels)
        self._upload_error = self._draw_error = None
        same = (keep_view and old is not None and volume is not None
                and np.allclose(old.extent, volume.extent, rtol=0.02))
        if volume is not None and not same:
            self.camera.preset("3D", volume.extent, max(self.width(), 1), max(self.height(), 1))
            self._fitted()
        self.cameraChanged.emit()
        self.update()

    def settings_changed(self) -> None:
        """Repaint after the caller edited channels / settings / camera in place (and try to draw again)."""
        self._draw_error = None
        self._burst()

    def set_preset(self, name: str) -> None:
        """A view preset (PRESETS) or "fit" (zoom to the box, rotation kept)."""
        if self.volume is None:
            return
        if name == "fit":
            self.camera.fit(self.volume.extent, self.width(), self.height())
        elif name in PRESETS:
            self.camera.preset(name, self.volume.extent, self.width(), self.height())
        else:
            raise ValueError(f"unknown view preset {name!r}")
        self._fitted()
        self.cameraChanged.emit()
        self.update()

    def set_message(self, text: str | None) -> None:
        """Text in the middle of the view (loading progress, errors); None removes it."""
        self._message = text or None
        self.update()

    def render_image(self, width: int | None = None, height: int | None = None) -> QtGui.QImage:
        """Full-quality image with overlays (default: the widget size in device pixels).

        Raises RuntimeError when OpenGL is not available or the image cannot be made.
        """
        dpr = self.devicePixelRatioF()
        lw, lh = max(self.width(), 1), max(self.height(), 1)
        w = int(width) if width else max(1, round(lw * dpr))
        h = int(height) if height else max(1, round(lh * dpr))
        if self._gl_error:
            raise RuntimeError(self._gl_error)
        if self.renderer.gl is None or self.context() is None or not self.context().isValid():
            off = OffscreenRenderer()      # never shown: no context of our own yet
            try:
                img = off.render(self.volume, self.channels, self.settings, self.camera, w, h)
            finally:
                off.release()
        else:
            self.makeCurrent()
            try:
                self._upload()
                img = self.renderer.render_to_image(self.channels, self.settings, self.camera, self.volume, w, h)
            finally:
                self.doneCurrent()
        return _overlay_image(img, self.camera, self.volume, self.settings, lw, lh)

    def max_texture_size(self) -> int:
        """Largest 3D texture side (DEFAULT_MAX_DIM until GL is up)."""
        return self.renderer.max_3d if self.renderer.gl is not None else DEFAULT_MAX_DIM

    def gl_info(self) -> str:
        """Renderer and OpenGL version, or the error text ("" before GL is initialised)."""
        return self._gl_error or self._info

    def release(self) -> None:
        """Drop the volume and its textures (the file was closed); channels and settings stay."""
        self.volume = None
        self._upload_error = self._draw_error = None
        if self.renderer.gl is not None and self.context() is not None and self.context().isValid():
            self.makeCurrent()
            self.renderer.release_volume()
            self.doneCurrent()
        self.update()

    # -- GL -----------------------------------------------------------------------------

    def initializeGL(self) -> None:
        ctx = self.context()
        ctx.aboutToBeDestroyed.connect(self._context_lost)
        self._gl_error = self.renderer.initialize(ctx)
        self._info = self.renderer.info
        self._full_ms = 0.0
        self._draw_error = None
        self.glReady.emit(self._gl_error is None, self._gl_error or self._info)

    def _context_lost(self) -> None:
        """Free the GL resources of a context that goes away (widget deleted or moved to another window)."""
        self.makeCurrent()
        if self._half_fbo is not None:
            sip.delete(self._half_fbo)
            self._half_fbo = None
        self.renderer.release()
        self.doneCurrent()

    def _fitted(self) -> None:
        """The camera was just fitted: re-fit on resize while its zoom and pan stay as they are.

        The panel edits the zoom (and the rotation) in place; a zoom typed there must not be
        replaced by the next fit.
        """
        self._auto_fit = True
        self._fit_state = (self.camera.zoom, tuple(self.camera.pan))

    def resizeGL(self, w: int, h: int) -> None:
        if self._auto_fit and self._fit_state != (self.camera.zoom, tuple(self.camera.pan)):
            self._auto_fit = False
        if self._auto_fit and self.volume is not None:
            self.camera.fit(self.volume.extent, self.width(), self.height())
            self._fitted()
            self.cameraChanged.emit()

    def _upload(self) -> None:
        t0 = time.perf_counter()
        self.renderer.upload(self.volume)
        dt = (time.perf_counter() - t0) * 1000.0
        if dt > 0.5:
            self.upload_ms = dt

    def paintGL(self) -> None:
        if self._gl_error:
            p = QtGui.QPainter(self)
            p.fillRect(self.rect(), QtGui.QColor(24, 24, 26))
            paint_overlays(p, self.camera, None, self.settings, self.width(), self.height(), self._gl_error)
            p.end()
            return
        size = self.size() * self.devicePixelRatioF()
        w, h = max(size.width(), 1), max(size.height(), 1)
        if self._upload_error is None:
            try:
                self._upload()
            except Exception as exc:   # out of graphics memory etc.: show it instead of the volume
                self._upload_error = f"The volume could not be shown: {exc}"
        volume = None if self._upload_error else self.volume
        if self._draw_error is None:
            try:
                self._draw(volume, w, h)
            except Exception as exc:   # a shader the driver cannot link etc.: show it, never raise from paintGL
                self._draw_error = f"The 3D view could not draw the volume: {exc}"
        p = QtGui.QPainter(self)
        if self._draw_error is not None:
            volume = None
            p.fillRect(self.rect(), _qcolor(NO_VOLUME_BG))
        paint_overlays(p, self.camera, volume, self.settings, self.width(), self.height(),
                       self._upload_error or self._draw_error or self._message)
        p.end()

    def _draw(self, volume: Volume | None, w: int, h: int) -> None:
        """Ray cast the frame (w x h device pixels): coarser passes while the view changes, else full quality."""
        interactive = self._press is not None or self._settle.isActive()
        gl = self.renderer.gl
        t0 = time.perf_counter()
        if interactive and self._full_ms > FAST_FRAME_MS and volume is not None:
            if self._full_ms > SLOW_FRAME_MS:
                self._draw_half(volume, w, h)
                self.last_pass = "half"
            else:
                self.renderer.draw(self.channels, self.settings, self.camera, volume, w, h, INTERACTIVE_STEP)
                self.last_pass = "interactive"
            gl.glFinish()
            self.last_frame_ms = (time.perf_counter() - t0) * 1000.0
        else:
            for y0 in range(0, h, STRIP_ROWS):   # strips of separate draw calls, like render_to_image
                self.renderer.draw(self.channels, self.settings, self.camera, volume, w, h,
                                   rows=(y0, min(STRIP_ROWS, h - y0)))
                gl.glFlush()
            gl.glFinish()
            self.last_frame_ms = (time.perf_counter() - t0) * 1000.0
            self.last_pass = "full"
            if volume is not None:
                self._full_ms = self.last_frame_ms

    def _draw_half(self, volume: Volume, w: int, h: int) -> None:
        """Interactive pass at half resolution through a framebuffer object, scaled up linearly."""
        hw, hh = max(1, w // 2), max(1, h // 2)
        fbo = self._half_fbo
        if fbo is None or fbo.width() != hw or fbo.height() != hh:
            if fbo is not None:
                sip.delete(fbo)
            fmt = QtGui.QOpenGLFramebufferObjectFormat()
            fmt.setAttachment(QtGui.QOpenGLFramebufferObject.NoAttachment)
            fmt.setInternalTextureFormat(GL_RGBA8)
            fbo = self._half_fbo = QtGui.QOpenGLFramebufferObject(hw, hh, fmt)
        if not fbo.isValid() or not fbo.bind():
            raise RuntimeError(f"OpenGL could not create a {hw} x {hh} image")
        try:
            self.renderer.draw(self.channels, self.settings, self.camera, volume, hw, hh, INTERACTIVE_STEP)
        finally:
            fbo.release()   # back to the widget's framebuffer
        self.renderer.blit(fbo.texture(), w, h)

    def _burst(self) -> None:
        """Coarse passes now, a full-quality pass SETTLE_MS after the last call."""
        self._settle.start()
        self.update()

    # -- mouse and keys -------------------------------------------------------------------

    def mousePressEvent(self, ev) -> None:
        if ev.button() not in (Qt.LeftButton, Qt.MiddleButton) or self.volume is None or self._press is not None:
            super().mousePressEvent(ev)
            return
        self._press = QtCore.QPointF(ev.localPos())
        self._press_camera = self.camera.copy()
        self._mode = ("pan" if ev.button() == Qt.MiddleButton or ev.modifiers() & Qt.ShiftModifier
                      else "rotate")
        self._auto_fit = False
        self.setCursor(Qt.ClosedHandCursor if self._mode == "pan" else Qt.SizeAllCursor)
        ev.accept()

    def mouseMoveEvent(self, ev) -> None:
        if self._press is None or self.volume is None:
            super().mouseMoveEvent(ev)
            return
        d = ev.localPos() - self._press
        dx, dy = d.x(), d.y()
        start = self._press_camera
        if self._mode == "rotate":
            if ev.modifiers() & (Qt.AltModifier | Qt.ControlModifier):   # one axis only
                if abs(dx) > abs(dy):
                    dy = 0.0
                else:
                    dx = 0.0
            cam = start.copy()
            cam.rotate_by(-ROT_SPEED * dx, ROT_SPEED * dy)
            self.camera.rotation = cam.rotation
        else:
            mu0 = Camera.base_um_per_px(self.volume.extent, self.width(), self.height())
            self.camera.pan = (start.pan[0] + dx * mu0, start.pan[1] + dy * mu0)
        self.update()
        ev.accept()

    def mouseReleaseEvent(self, ev) -> None:
        if self._press is None or ev.button() not in (Qt.LeftButton, Qt.MiddleButton):
            super().mouseReleaseEvent(ev)
            return
        self.mouseMoveEvent(ev)
        self._press = self._press_camera = None
        self.unsetCursor()
        self._burst()
        self.cameraChanged.emit()

    def wheelEvent(self, ev) -> None:
        steps = ev.angleDelta().y() / 120.0
        if self.volume is None or not steps:
            super().wheelEvent(ev)
            return
        pos = ev.posF()
        self.camera.zoom_about(pos.x(), pos.y(), WHEEL_STEP ** steps, self.volume.extent, self.width(), self.height())
        self._auto_fit = False
        self._burst()
        self.cameraChanged.emit()
        ev.accept()

    def _handles(self, ev) -> bool:
        mods = ev.modifiers() & ~(Qt.ShiftModifier | Qt.KeypadModifier)
        return ev.key() in self._KEYS and not mods

    def event(self, ev) -> bool:
        # keep the main window's single-key shortcuts (F, ...) from taking our keys while we have focus
        if ev.type() == QtCore.QEvent.ShortcutOverride and self._handles(ev):
            ev.accept()
            return True
        return super().event(ev)

    def keyPressEvent(self, ev) -> None:
        if not self._handles(ev) or self.volume is None:
            super().keyPressEvent(ev)
            return
        key, step = ev.key(), (1.0 if ev.modifiers() & Qt.ShiftModifier else 5.0)
        if key in (Qt.Key_R, Qt.Key_Home):
            self.set_preset("3D")
        elif key == Qt.Key_F:
            self.set_preset("fit")
        elif key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            # like a drag in the arrow's direction: the front follows
            ax = {Qt.Key_Left: step, Qt.Key_Right: -step}.get(key, 0.0)
            ay = {Qt.Key_Up: -step, Qt.Key_Down: step}.get(key, 0.0)
            self.camera.rotate_by(ax, ay)
            self._auto_fit = False
            self._burst()
            self.cameraChanged.emit()
        else:
            factor = 1.0 / WHEEL_STEP if key == Qt.Key_Minus else WHEEL_STEP
            self.camera.zoom_about(self.width() / 2.0, self.height() / 2.0, factor, self.volume.extent,
                                   self.width(), self.height())
            self._auto_fit = False
            self._burst()
            self.cameraChanged.emit()
        ev.accept()


# -----------------------------------------------------------------------------
# loading
# -----------------------------------------------------------------------------

class VolumeLoadThread(QtCore.QThread):
    """msr_volume.load_volume in the background (no GL here: the view uploads in the GUI thread)."""

    progress = QtCore.pyqtSignal(int, int)    # planes read, planes in total
    done = QtCore.pyqtSignal(object)          # Volume, or None when cancelled
    failed = QtCore.pyqtSignal(str)           # "ExceptionType: message"

    def __init__(self, sources: list, index: tuple, *, max_dim: int = DEFAULT_MAX_DIM, budget: int = DEFAULT_BUDGET,
                 factor: tuple[int, int, int] | None = None, key=None, parent=None):
        super().__init__(parent)
        self.sources, self.index = list(sources), tuple(index)
        self.max_dim, self.budget, self.factor = max_dim, budget, factor
        self.key = key                 # the caller's identifier of this load
        self._cancel = False
        self._last_emit = 0.0

    def cancel(self) -> None:
        self._cancel = True

    def _progress(self, done: int, total: int) -> None:
        now = time.monotonic()
        if done >= total or now - self._last_emit > 0.05:
            self._last_emit = now
            self.progress.emit(done, total)

    def run(self) -> None:
        try:
            volume = load_volume(self.sources, self.index, max_dim=self.max_dim, budget=self.budget,
                                 factor=self.factor, progress=self._progress, cancelled=lambda: self._cancel)
            self.done.emit(None if self._cancel else volume)
        except InterruptedError:
            self.done.emit(None)
        except Exception as exc:  # shown by the caller
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.sources = []      # frozen mosaics hold a plane cache: let it go
