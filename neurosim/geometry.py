"""Triangle meshes: import (STL ascii/binary, OBJ, PLY), built-in parametric models, placement, export.

A mesh is a float32 array of shape (n, 3, 3): n triangles x 3 vertices x xyz.
"""
import struct
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ import / export


def load(path):
    path = Path(path)
    ext = path.suffix.lower()
    data = path.read_bytes()
    if ext == ".stl":
        return _stl(data)
    if ext == ".obj":
        return _obj(data.decode("utf-8", "replace"))
    if ext == ".ply":
        return _ply(data)
    raise ValueError(f"unsupported mesh format: {ext} (use .stl, .obj or .ply)")


def _stl(data):
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + 50 * n:  # binary
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
            return rec["v"].astype(np.float32)
    text = data.decode("utf-8", "replace")
    v = [list(map(float, line.split()[1:4])) for line in text.splitlines() if line.strip().startswith("vertex")]
    return np.asarray(v, np.float32).reshape(-1, 3, 3)


def _obj(text):
    verts, tris = [], []
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            verts.append([float(p[1]), float(p[2]), float(p[3])])
        elif p[0] == "f":
            idx = [int(q.split("/")[0]) for q in p[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            tris += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]  # fan triangulation
    return np.asarray(verts, np.float32)[np.asarray(tris)]


def _ply(data):
    end = data.index(b"end_header") + len(b"end_header")
    end += 2 if data[end:end + 2] == b"\r\n" else 1
    header = data[:end].decode("ascii", "replace").splitlines()
    fmt = next(l.split()[1] for l in header if l.startswith("format"))
    elements, cur = [], None
    for l in header:
        p = l.split()
        if p and p[0] == "element":
            cur = [p[1], int(p[2]), []]
            elements.append(cur)
        elif p and p[0] == "property":
            cur[2].append(p[1:])
    types = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
             "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2", "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}
    body = data[end:]
    verts = faces = None
    if fmt == "ascii":
        tokens = body.split()
        pos = 0
        for name, count, props in elements:
            rows = []
            for _ in range(count):
                if props and props[0][0] == "list":
                    k = int(tokens[pos]); rows.append([int(t) for t in tokens[pos + 1:pos + 1 + k]]); pos += 1 + k
                else:
                    rows.append([float(t) for t in tokens[pos:pos + len(props)]]); pos += len(props)
            if name == "vertex":
                names = [p[-1] for p in props]
                verts = np.asarray(rows, np.float32)[:, [names.index("x"), names.index("y"), names.index("z")]]
            elif name == "face":
                faces = rows
    else:
        bo = "<" if "little" in fmt else ">"
        pos = 0
        for name, count, props in elements:
            if props and props[0][0] == "list":
                ct, it = types[props[0][1]], types[props[0][2]]
                rows = []
                for _ in range(count):
                    k = int(np.frombuffer(body, bo + ct, 1, pos)[0]); pos += np.dtype(ct).itemsize
                    rows.append(np.frombuffer(body, bo + it, k, pos).tolist()); pos += k * np.dtype(it).itemsize
                if name == "face":
                    faces = rows
            else:
                dt = np.dtype([(p[-1], bo + types[p[0]]) for p in props])
                arr = np.frombuffer(body, dt, count, pos); pos += count * dt.itemsize
                if name == "vertex":
                    verts = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float32)
    tris = [[f[0], f[k], f[k + 1]] for f in faces for k in range(1, len(f) - 1)]
    return verts[np.asarray(tris)]


def save_stl(tris, path):
    tris = np.ascontiguousarray(tris, np.float32)
    rec = np.zeros(len(tris), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["v"] = tris
    with open(path, "wb") as f:
        f.write(b"NeuroSim binary STL".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        f.write(rec.tobytes())


# ------------------------------------------------------------------ placement


def rotation(deg):
    ax, ay, az = np.radians(deg)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def place(tris, size, center, rotation_deg=(0, 0, 0)):
    """Rotate about the bounding-box center, scale the longest side to `size`, move the center to `center`."""
    t = tris.reshape(-1, 3).astype(np.float64)
    c = 0.5 * (t.min(0) + t.max(0))
    t = (t - c) @ rotation(rotation_deg).T
    extent = (t.max(0) - t.min(0)).max()
    t = t * (size / extent if extent > 0 else 1.0)
    t += np.asarray(center) - 0.5 * (t.min(0) + t.max(0))
    return t.reshape(-1, 3, 3).astype(np.float32)


def frontal_area(tris, axis, resolution=1.0):
    """Projected area (in cells^2) of the mesh seen along `axis` (0/1/2), by rasterizing the projected triangles."""
    a, b = [i for i in range(3) if i != axis]
    p = tris[:, :, [a, b]].astype(np.float64)
    lo = np.floor(p.reshape(-1, 2).min(0)) - 1
    hi = np.ceil(p.reshape(-1, 2).max(0)) + 1
    nx, ny = ((hi - lo) / resolution).astype(int) + 1
    mask = np.zeros((nx, ny), bool)
    for tri in (p - lo) / resolution:
        x0, y0 = np.floor(tri.min(0)).astype(int)
        x1, y1 = np.ceil(tri.max(0)).astype(int)
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5, indexing="ij")
        (ax, ay), (bx, by), (cx, cy) = tri
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(d) < 1e-12:
            continue
        l1 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / d
        l2 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / d
        inside = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
        mask[x0:x1 + 1, y0:y1 + 1] |= inside[: nx - x0, : ny - y0]
    return float(mask.sum()) * resolution ** 2


# ------------------------------------------------------------------ built-in models


def _grid_surface(fn, nu, nv, wrap_u=True):
    """Triangulate a parametric surface fn(u, v) -> xyz on [0,1]^2."""
    u = np.linspace(0, 1, nu + (0 if wrap_u else 1), endpoint=not wrap_u)
    v = np.linspace(0, 1, nv + 1)
    P = fn(*np.meshgrid(u, v, indexing="ij"))  # (nu, nv+1, 3)
    iu = np.arange(len(u))
    iu1 = (iu + 1) % len(u) if wrap_u else iu + 1
    if not wrap_u:
        iu, iu1 = iu[:-1], iu1[:-1]
    tris = []
    for j in range(nv):
        a, b, c, d = P[iu, j], P[iu1, j], P[iu1, j + 1], P[iu, j + 1]
        tris.append(np.stack([a, b, c], 1))
        tris.append(np.stack([a, c, d], 1))
    return np.concatenate(tris).astype(np.float32)


def _cap(ring, center):
    return np.stack([np.repeat(center[None], len(ring), 0), np.roll(ring, -1, 0), ring], 1).astype(np.float32)


def sphere(n=48):
    def f(u, v):
        th, ph = 2 * np.pi * u, np.pi * v
        return np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)], -1)
    return _grid_surface(f, n, n // 2)


def cylinder(n=64, height=2.0):
    side = _grid_surface(lambda u, v: np.stack([np.cos(2 * np.pi * u), np.sin(2 * np.pi * u), height * (v - 0.5)], -1), n, 1)
    th = 2 * np.pi * np.arange(n) / n
    ring = np.stack([np.cos(th), np.sin(th), np.zeros(n)], 1)
    top, bot = ring + [0, 0, height / 2], ring - [0, 0, height / 2]
    return np.concatenate([side, _cap(top, np.array([0, 0, height / 2]))[:, ::-1], _cap(bot, np.array([0, 0, -height / 2]))])


def box(lx=1.0, ly=1.0, lz=1.0):
    v = np.array([[x, y, z] for x in (-lx / 2, lx / 2) for y in (-ly / 2, ly / 2) for z in (-lz / 2, lz / 2)])
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([[v[a], v[b], v[c]] for a, b, c, d in q] + [[v[a], v[c], v[d]] for a, b, c, d in q], np.float32)


def torus(R=1.0, r=0.35, n=64, m=24):
    def f(u, v):
        th, ph = 2 * np.pi * u, 2 * np.pi * v
        return np.stack([(R + r * np.cos(ph)) * np.cos(th), (R + r * np.cos(ph)) * np.sin(th), r * np.sin(ph)], -1)
    return _grid_surface(f, n, m)


def naca_profile(code="0012", n=80):
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    if m > 0:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x ** 2))
        dy = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc, dy = np.zeros_like(x), np.zeros_like(x)
    th = np.arctan(dy)
    upper = np.stack([x - yt * np.sin(th), yc + yt * np.cos(th)], 1)
    lower = np.stack([x + yt * np.sin(th), yc - yt * np.cos(th)], 1)
    return np.concatenate([upper[::-1], lower[1:-1]])  # closed loop, trailing edge -> leading edge -> trailing edge


def wing(code="0012", span=3.0):
    """Straight wing: chord along x (flow), thickness along z, span along y."""
    prof = naca_profile(code)
    k = len(prof)
    a = np.stack([prof[:, 0], np.full(k, -span / 2), prof[:, 1]], 1)
    b = a + [0, span, 0]
    tris = []
    for i in range(k):
        j = (i + 1) % k
        tris += [[a[i], a[j], b[j]], [a[i], b[j], b[i]]]
    ca, cb = a.mean(0), b.mean(0)
    tris += [[ca, a[(i + 1) % k], a[i]] for i in range(k)] + [[cb, b[i], b[(i + 1) % k]] for i in range(k)]
    return np.asarray(tris, np.float32)


def _loft(sections):
    """Close a tube through a list of rings (each (k,3)), with end caps."""
    tris = []
    for s0, s1 in zip(sections[:-1], sections[1:]):
        k = len(s0)
        for i in range(k):
            j = (i + 1) % k
            tris += [[s0[i], s0[j], s1[j]], [s0[i], s1[j], s1[i]]]
    tris += _cap(sections[0], sections[0].mean(0))[:, ::-1].tolist() + _cap(sections[-1], sections[-1].mean(0)).tolist()
    return np.asarray(tris, np.float32)


def _rounded_rect(w, h, r, n=8):
    pts = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for a in np.radians(np.linspace(a0, a0 + 90, n)):
            pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    return np.asarray(pts)


def ahmed_body(slant_deg=25.0):
    """Ahmed reference body (flow along +x): 1.044 x 0.389 x 0.288, rounded nose, slanted rear, no stilts."""
    L, W, H, R = 1.044, 0.389, 0.288, 0.1
    cs = _rounded_rect(W, H, 0.05)
    secs = []
    for x in np.linspace(0, R, 8):  # nose rounding radius 0.1 in side and top views
        s = R - np.sqrt(max(R ** 2 - (R - x) ** 2, 0))
        shrink = np.clip(1 - 2 * s / np.array([W, H]), 0.05, 1)
        secs.append(np.c_[np.full(len(cs), x), cs * shrink])
    slant_len = 0.222
    drop = slant_len * np.tan(np.radians(slant_deg))
    for x in np.linspace(R, L, 12)[1:]:
        ring = np.c_[np.full(len(cs), x), cs.copy()]
        if x > L - slant_len:  # slanted rear: lower the roof towards the tail
            top = H / 2 - drop * (x - (L - slant_len)) / slant_len
            ring[:, 2] = np.minimum(ring[:, 2], top)
        secs.append(ring)
    return _loft([s[:, [0, 1, 2]] for s in secs])


def car():
    """Simple passenger-car shape (flow along +x): lofted body with a cabin and four wheels."""
    xs = np.linspace(0, 4.2, 24)
    roof = np.interp(xs, [0, 0.3, 1.2, 1.7, 3.2, 3.9, 4.2], [0.55, 0.85, 0.95, 1.4, 1.42, 1.0, 0.8])
    bottom = np.interp(xs, [0, 0.3, 3.9, 4.2], [0.35, 0.25, 0.25, 0.45])
    width = np.interp(xs, [0, 0.4, 3.8, 4.2], [1.5, 1.75, 1.75, 1.6])
    secs = []
    for x, zt, zb, w in zip(xs, roof, bottom, width):
        cs = _rounded_rect(w, zt - zb, min(0.2, 0.45 * (zt - zb)))
        secs.append(np.c_[np.full(len(cs), x), cs[:, 0], cs[:, 1] + 0.5 * (zt + zb)])
    body = _loft(secs)
    wheel = cylinder(32, height=0.25) * [0.33, 0.33, 1.0]
    wheel = wheel[:, :, [0, 2, 1]]  # axle along y
    wheels = [wheel + [x, y, 0.33] for x in (0.85, 3.35) for y in (-0.78, 0.78)]
    return np.concatenate([body] + wheels).astype(np.float32)


def rotor(blades=4, twist_deg=30.0, hub=0.18):
    """Fan/propeller in the xy plane, axis along z."""
    parts = [cylinder(48, height=0.3) * [hub, hub, 1.0]]
    for k in range(blades):
        blade = []
        for r in np.linspace(hub * 0.9, 1.0, 10):
            prof = naca_profile("4412", 24) - [0.5, 0]
            chord = 0.32 * (1.2 - 0.6 * r)
            a = np.radians(twist_deg * (1.1 - r) + 10)
            x = chord * (prof[:, 0] * np.cos(a) - prof[:, 1] * np.sin(a))
            z = chord * (prof[:, 0] * np.sin(a) + prof[:, 1] * np.cos(a))
            blade.append(np.stack([x, np.full(len(x), r), z], 1))
        tris = _loft(blade)
        c, s = np.cos(2 * np.pi * k / blades), np.sin(2 * np.pi * k / blades)
        parts.append(tris @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], np.float32))
    return np.concatenate(parts).astype(np.float32)


BUILTIN = {
    "sphere": ("Sphere", sphere),
    "cube": ("Cube", box),
    "cylinder": ("Cylinder", cylinder),
    "torus": ("Torus", torus),
    "naca0012": ("Wing NACA 0012", lambda: wing("0012")),
    "naca4412": ("Wing NACA 4412", lambda: wing("4412")),
    "ahmed25": ("Ahmed body 25°", ahmed_body),
    "car": ("Car", car),
    "fan4": ("Fan, 4 blades", rotor),
    "prop3": ("Propeller, 3 blades", lambda: rotor(3, 40.0, 0.12)),
}


def builtin(name):
    return BUILTIN[name][1]().astype(np.float32)


if __name__ == "__main__":  # self-check: round trips and areas
    import tempfile
    s = place(sphere(96), 40.0, (50, 50, 50))
    a = frontal_area(s, 0)
    assert abs(a / (np.pi * 20 ** 2) - 1) < 0.05, a
    with tempfile.TemporaryDirectory() as d:
        save_stl(s, Path(d) / "s.stl")
        assert np.allclose(load(Path(d) / "s.stl"), s)
        (Path(d) / "b.obj").write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
        assert load(Path(d) / "b.obj").shape == (2, 3, 3)
        (Path(d) / "t.ply").write_text("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
        assert load(Path(d) / "t.ply").shape == (1, 3, 3)
    for k in BUILTIN:
        t = builtin(k)
        assert t.ndim == 3 and t.shape[1:] == (3, 3) and np.isfinite(t).all(), k
    print("geometry ok", {k: len(builtin(k)) for k in BUILTIN})
