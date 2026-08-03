"""Optional Weights & Biases tracking.

A thin wrapper so the training loop can call ``run.log(...)`` unconditionally.
When ``--wandb`` is not passed (or wandb is not installed) every call is a no-op,
so nothing about the default behaviour changes.

Cluster note: FASRC compute nodes may not have outbound internet. If an online
init fails we retry in ``offline`` mode, which writes to ``<run_dir>/wandb/`` and
can be uploaded later from a login node with ``wandb sync <dir>``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class JsonlMirror:
    """Append every logged payload to <run_dir>/history.jsonl.

    Diagnostics used to live only in wandb, so reading back per-group
    update-to-weight ratios (to set per-part learning rates) or per-head val
    curves needed API access to a service the compute nodes cannot always
    reach. The run directory should be self-describing: one JSON object per
    log call, in order.
    """

    def __init__(self, run_dir: Path | None) -> None:
        self._path = Path(run_dir) / "history.jsonl" if run_dir else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, metrics: dict[str, Any], step: int | None) -> None:
        if self._path is None:
            return
        row = {"_step": step, **{k: v for k, v in metrics.items()
                                 if isinstance(v, (int, float, str, bool, type(None)))}}
        try:
            with self._path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError:
            pass          # never let telemetry kill a training run


class NullRun:
    """Stand-in used when tracking is disabled. Still mirrors to history.jsonl,
    so a run without wandb is just as analysable afterwards."""

    enabled = False

    def __init__(self, mirror: JsonlMirror | None = None) -> None:
        self._mirror = mirror or JsonlMirror(None)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._mirror.write(metrics, step)
        return None

    def summary(self, metrics: dict[str, Any]) -> None:
        return None

    def log_table(self, name: str, dataframe: Any) -> None:
        return None

    def finish(self) -> None:
        return None


class WandbRun:
    """Live wandb run. Constructed only via :func:`init_tracking`."""

    enabled = True

    def __init__(self, wandb_module: Any, run: Any, mirror: JsonlMirror | None = None) -> None:
        self._wandb = wandb_module
        self._run = run
        self._mirror = mirror or JsonlMirror(None)

    @property
    def url(self) -> str:
        return getattr(self._run, "url", "") or "(offline)"

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._mirror.write(metrics, step)
        self._wandb.log(metrics, step=step)

    def summary(self, metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            self._run.summary[key] = value

    def log_table(self, name: str, dataframe: Any) -> None:
        self._wandb.log({name: self._wandb.Table(dataframe=dataframe)})

    def finish(self) -> None:
        self._wandb.finish()


def init_tracking(
    *,
    enabled: bool,
    project: str,
    entity: str | None,
    run_name: str | None,
    mode: str,
    config: dict[str, Any],
    run_dir: Path,
    tags: list[str] | None = None,
) -> NullRun | WandbRun:
    """Return a live :class:`WandbRun` or a :class:`NullRun` fallback.

    Never raises: a tracking failure must not take down a training job.
    """
    if not enabled:
        return NullRun(JsonlMirror(run_dir))

    try:
        import wandb
    except ImportError:
        print("[wandb] not installed (pip install wandb) - continuing without tracking", flush=True)
        return NullRun(JsonlMirror(run_dir))

    # Keep all wandb state inside the run directory so it lands on the shared FS
    # next to the checkpoints, not in the submit directory.
    os.environ.setdefault("WANDB_DIR", str(run_dir))

    def _init(selected_mode: str) -> Any:
        return wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            mode=selected_mode,
            config=config,
            dir=str(run_dir),
            tags=tags or [],
            reinit=True,
        )

    try:
        run = _init(mode)
    except Exception as exc:  # network down on a compute node, bad API key, ...
        if mode == "online":
            print(f"[wandb] online init failed ({exc}); falling back to offline", flush=True)
            try:
                run = _init("offline")
            except Exception as exc2:
                print(f"[wandb] offline init also failed ({exc2}); continuing without tracking", flush=True)
                return NullRun(JsonlMirror(run_dir))
        else:
            print(f"[wandb] init failed ({exc}); continuing without tracking", flush=True)
            return NullRun(JsonlMirror(run_dir))

    tracked = WandbRun(wandb, run)
    print(f"[wandb] tracking run -> {tracked.url}", flush=True)
    return tracked
