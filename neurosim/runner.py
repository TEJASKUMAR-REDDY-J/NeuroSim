"""Simulation runs: one run directory + one solver worker process per run."""
import datetime
import json
import os
import platform
import re
import subprocess
import threading
from pathlib import Path

from . import build, case as case_mod

WORKSPACE = build.ROOT / "workspace"
RUNS = WORKSPACE / "runs"


def _git(*args, cwd=build.ROOT):
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    return r.stdout.strip()


def provenance(variant):
    return {
        "neurosim_commit": _git("rev-parse", "HEAD"),
        "neurosim_dirty": bool(_git("status", "--porcelain", "--", "neurosim", "solver")),
        "fluidx3d_commit": build.fluidx3d_commit(),
        "patches": [p.name for p in build._patches()],
        "variant": variant,
        "host": {"os": platform.platform(), "cpu": platform.processor(), "python": platform.python_version()},
    }


class Run:
    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        self.case = json.loads((self.dir / "case.json").read_text())
        self.meta = json.loads((self.dir / "run.json").read_text())
        self.proc = None
        self.status = self.meta.get("status", "stopped")
        self.records = []
        self._offset = 0
        self.error = None

    @property
    def id(self):
        return self.dir.name

    @classmethod
    def create(cls, case, restart=None, root=RUNS):
        case = dict(case)
        if restart:
            case["restart"] = str(restart)
        run_dir = new_dir(case.get("name", "run"), root)
        variant, derived, full = case_mod.prepare(case, run_dir)
        (run_dir / "case.json").write_text(json.dumps(full, indent=1))
        meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "status": "created", "derived": derived, **provenance(variant)}
        (run_dir / "run.json").write_text(json.dumps(meta, indent=1, default=float))
        return cls(run_dir)

    def _save_meta(self, **kw):
        self.meta.update(kw)
        (self.dir / "run.json").write_text(json.dumps(self.meta, indent=1, default=float))

    def start(self, log=print, paused=False):
        """Builds the solver variant if needed (slow the first time) and launches the worker."""
        self.status = "building"
        exe = build.build(self.meta["variant"], log)
        cfg = self.dir / "case.cfg"
        text = cfg.read_text()
        text = re.sub(r"^start_paused = .*\n", "", text, flags=re.M) + f"start_paused = {'true' if paused else 'false'}\n"
        cfg.write_text(text)
        flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        self._log = open(self.dir / "solver.log", "a")
        self.proc = subprocess.Popen([str(exe), str(cfg), str(self.dir)], cwd=self.dir, stdin=subprocess.PIPE, stdout=self._log, stderr=subprocess.STDOUT, text=True, creationflags=flags)
        self.status = "paused" if paused else "starting"
        self._save_meta(status="running", started=datetime.datetime.now().isoformat(timespec="seconds"), executable=str(exe))

    def send(self, command):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(command + "\n")
                self.proc.stdin.flush()
                if command == "pause":
                    self.status = "paused"
                elif command == "resume":
                    self.status = "running"
            except OSError:
                pass

    def poll(self):
        """New telemetry records since the last call; also tracks process state."""
        path = self.dir / "telemetry.jsonl"
        new = []
        if path.exists():
            with open(path, "rb") as f:
                f.seek(self._offset)
                chunk = f.read()
            end = chunk.rfind(b"\n") + 1  # only complete lines
            self._offset += end
            for line in chunk[:end].splitlines():
                try:
                    new.append(json.loads(line))
                except ValueError:
                    pass
        for r in new:
            t = r.get("type")
            if t == "start" and self.status == "starting":
                self.status = "running"
            elif t == "finished":
                self.status = "finished"
            elif t == "error":
                self.error = r.get("message")
        self.records += new
        if self.proc and self.proc.poll() is not None and self.status not in ("stopped", "failed"):
            code = self.proc.returncode
            self.status = "stopped" if code == 0 else "failed"
            if code:
                self.error = self.error or self._last_error() or f"solver exited with code {code}"
            self._log.close()
            self._save_meta(status=self.status, ended=datetime.datetime.now().isoformat(timespec="seconds"), error=self.error)
        elif self.proc and self.proc.poll() is None and self.status in ("starting", "running"):
            err = self._last_error()
            if err:  # FluidX3D waits for Enter after an error on Windows; do not hang
                self.error = err
                self.proc.kill()
        return new

    def _last_error(self):
        p = self.dir / "solver.log"
        if not p.exists():
            return None
        lines = [l for l in p.read_text(errors="replace").splitlines() if "Error" in l]
        return lines[-1].strip("| ").strip() if lines else None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.send("stop")
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.poll()

    def load_records(self):
        """For finished runs: read all telemetry from disk."""
        self.records, self._offset = [], 0
        return self.poll()

    def checkpoints(self):
        return sorted((self.dir / "checkpoints").glob("*.nsck"), key=lambda p: int(p.stem[1:]))


def new_dir(name, root=RUNS):
    """Unique, sortable history directory: <stamp>-<slug>."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "run"
    d, n = Path(root) / f"{stamp}-{slug}", 1
    while d.exists():
        d = Path(root) / f"{stamp}-{slug}-{n}"; n += 1
    return d


def list_runs():
    return sorted([p for p in RUNS.glob("*") if (p / "run.json").exists()], reverse=True)


def devices():
    exe = build.build(build.variant())
    r = subprocess.run([str(exe), "--devices"], capture_output=True, text=True, timeout=120, creationflags=0x08000000 if os.name == "nt" else 0)
    return json.loads(r.stdout.strip().splitlines()[-1])
