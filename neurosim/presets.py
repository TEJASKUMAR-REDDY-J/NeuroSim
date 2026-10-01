"""Ready-to-run cases covering FluidX3D's capabilities. Sizes target a few million cells so they run on
integrated GPUs; raise `domain` for larger GPUs."""

FS = "freestream"


def _tunnel(ground=None):
    b = {f: FS for f in ("x0", "x1", "y0", "y1", "z0", "z1")}
    if ground:
        b["z0"] = ground
    return b


PRESETS = {
    "Sphere in wind tunnel (drag)": {
        "domain": [128, 256, 128], "flow": {"direction": "+y", "u": 0.075, "re": 1000},
        "boundaries": _tunnel(), "forces": True, "les": True,
        "objects": [{"model": "sphere", "size": 32, "position": [0.5, 0.3, 0.5]}],
        "view": {"modes": ["lattice", "surface", "q_criterion"], "slice_axis": 0},
    },
    "Cylinder: Karman vortex street": {
        "domain": [96, 384, 24], "flow": {"direction": "+y", "u": 0.075, "re": 250},
        "boundaries": {"x0": FS, "x1": FS, "y0": FS, "y1": FS, "z0": "periodic", "z1": "periodic"}, "forces": True,
        "objects": [{"shape": "cylinder", "radius": 10, "axis": [0, 0, 1], "position": [0.5, 0.2, 0.5]}],
        "view": {"modes": ["lattice", "surface", "field"], "field": 0, "slice_axis": 2, "slice_field": 5, "camera": [0, 89, 60, 1.6]},
    },
    "Car on moving road": {
        "domain": [128, 352, 96], "flow": {"direction": "+y", "u": 0.075, "re": 100000},
        "boundaries": _tunnel(ground="moving"), "wall_velocity": [0, 0.075, 0], "forces": True, "les": True,
        "objects": [{"model": "car", "size": 150, "rotation": [0, 0, 90], "position": [0.5, 0.35, 0.3], "on_ground": True}],
        "view": {"modes": ["lattice", "surface", "q_criterion"], "slice_axis": 0, "camera": [-35, 25, 60, 1.2]},
    },
    "Ahmed body 25 deg": {
        "domain": [112, 320, 80], "flow": {"direction": "+y", "u": 0.075, "re": 50000},
        "boundaries": _tunnel(ground="wall"), "forces": True, "les": True,
        "objects": [{"model": "ahmed25", "size": 104, "rotation": [0, 0, 90], "position": [0.5, 0.3, 0.3], "on_ground": True}],
        "view": {"modes": ["lattice", "surface", "q_criterion"], "slice_axis": 0},
    },
    "Wing NACA 4412 at 8 deg": {
        "domain": [192, 256, 96], "flow": {"direction": "+y", "u": 0.075, "re": 20000},
        "boundaries": _tunnel(), "forces": True, "les": True, "reference_area": "planform",
        "objects": [{"model": "naca4412", "size": 150, "rotation": [0, 8, 90], "position": [0.5, 0.3, 0.5]}],
        "view": {"modes": ["lattice", "surface", "q_criterion"], "slice_axis": 0},
    },
    "Rotating fan in a box": {
        "domain": [160, 160, 112], "flow": {"direction": "+z", "u": 0.1, "re": 50000, "length": 90},
        "boundaries": {f: "wall" for f in ("x0", "x1", "y0", "y1", "z0", "z1")}, "init": "rest", "les": True,
        "objects": [{"model": "fan4", "size": 96, "position": [0.5, 0.5, 0.45], "tip_speed": 0.1, "spin_axis": [0, 0, 1]}],
        "view": {"modes": ["lattice", "surface", "q_criterion"]},
    },
    "Lid-driven cavity (Re 1000)": {
        "domain": [128, 128, 128], "flow": {"direction": "+x", "u": 0.1, "re": 1000, "length": 126}, "init": "rest",
        "boundaries": {"x0": "wall", "x1": "wall", "y0": "wall", "y1": "wall", "z0": "wall", "z1": "moving"}, "wall_velocity": [0.1, 0, 0],
        "view": {"modes": ["lattice", "streamlines"], "slice_axis": 1},
    },
    "Lid-driven cavity with tracer particles": {
        "domain": [96, 96, 96], "flow": {"direction": "+x", "u": 0.1, "re": 1000, "length": 94}, "init": "rest",
        "boundaries": {"x0": "wall", "x1": "wall", "y0": "wall", "y1": "wall", "z0": "wall", "z1": "moving"}, "wall_velocity": [0.1, 0, 0],
        "particles": {"count": 20000, "rho": 1.0},
        "view": {"modes": ["lattice", "particles"], "slice_axis": 1},
    },
    "Dam break (free surface)": {
        "domain": [96, 192, 128], "nu": 0.005, "flow": {"direction": "+y", "u": 0.05, "re": 1000}, "init": "rest",
        "boundaries": {f: "wall" for f in ("x0", "x1", "y0", "y1", "z0", "z1")}, "gravity": [0, 0, -0.0002], "surface_tension": 0.0001,
        "fills": [[0, 0, 0, 1, 0.25, 0.75]],
        "view": {"modes": ["raytrace"], "slice_axis": 0, "slice_field": 6, "camera": [40, 20, 60, 1.1]},
    },
    "Rayleigh-Benard convection": {
        "domain": [192, 192, 48], "nu": 0.02, "flow": {"direction": "+z", "u": 0.05, "re": 100}, "init": "rest", "init_noise": 0.015,
        "boundaries": {"x0": "periodic", "x1": "periodic", "y0": "periodic", "y1": "periodic", "z0": "hot", "z1": "cold"},
        "gravity": [0, 0, -0.0005], "thermal": {"alpha": 1.0, "beta": 1.0, "T_hot": 1.75, "T_cold": 0.25},
        "view": {"modes": ["lattice", "streamlines"], "field": 2, "slice_axis": 2, "slice_field": 6},
    },
    "Taylor-Green vortex (validation)": {
        "domain": [128, 128, 128], "nu": 0.01, "flow": {"direction": "+x", "u": 0.05, "re": 100}, "init": "taylor_green",
        "view": {"modes": ["streamlines"], "slice_axis": 2, "slice_field": 5},
    },
}
