"""FitChip Orchestrator — FastAPI service in front of the core pipeline.

Two lanes, per the architecture doc:

- fast lane  (sync):  /v1/inspect — parse + validate + estimate, a few hundred
  ms, so a GUI can react before the user hits "compile".
- slow lane:          /v1/compile — runs the real compilation. MVP executes
  in-process and synchronously; wave 2 (together with the TVM backend) moves
  this behind Celery+Redis and into per-backend ephemeral Docker containers.
  The API shape (job id in the response) is already async-friendly.

Run:  uvicorn fitchip.orchestrator.app:app --reload      (pip install 'fitchip[server]')
"""

from __future__ import annotations

import dataclasses
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The orchestrator requires the server extra: pip install 'fitchip[server]'"
    ) from exc

import fitchip
from fitchip.core.pipeline import Pipeline
from fitchip.core.selection.engine import SelectionReport

app = FastAPI(
    title="FitChip Orchestrator",
    version=fitchip.__version__,
    description="Compile trained ML models into ready-to-flash firmware projects.",
)

_pipeline = Pipeline()
# MVP job store: compiled artifacts kept on disk until the process exits.
# Wave 2 replaces this with the Celery result backend.
_jobs: dict[str, dict] = {}


@app.get("/v1/health")
def health() -> dict:
    return {"status": "ok", "version": fitchip.__version__}


@app.get("/v1/targets")
def targets() -> list[dict]:
    return [dataclasses.asdict(t) for t in _pipeline.targets.all()]


@app.get("/v1/backends")
def backends() -> list[dict]:
    return [b.capabilities() for b in _pipeline.backends.all()]


@app.post("/v1/inspect")
async def inspect(
    model: UploadFile = File(...),
    target: str = Form(...),
    quantize: str | None = Form(None),
) -> dict:
    """Fast lane: compatibility + memory report, no compilation."""
    with tempfile.TemporaryDirectory(prefix="fitchip-inspect-") as tmp:
        model_path = Path(tmp) / (model.filename or "model")
        model_path.write_bytes(await model.read())
        try:
            req = _pipeline.build_request(
                str(model_path), target, quantize="int8_full" if quantize == "int8" else None
            )
            meta, selection = _pipeline.inspect(req)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"model": meta.to_dict(), "selection": _selection_dict(selection)}


@app.post("/v1/compile")
async def compile_model(
    model: UploadFile = File(...),
    target: str = Form(...),
    quantize: str | None = Form(None),
    optimize_for: str = Form("size"),
    backend: str | None = Form(None),
) -> dict:
    """Slow lane. Synchronous in the MVP; the response already carries a
    job id so clients built against it survive the move to async."""
    job_id = uuid.uuid4().hex
    job_dir = Path(tempfile.mkdtemp(prefix=f"fitchip-job-{job_id}-"))
    model_path = job_dir / (model.filename or "model")
    model_path.write_bytes(await model.read())

    try:
        req = _pipeline.build_request(
            str(model_path),
            target,
            quantize="int8_full" if quantize == "int8" else None,
            optimize_for=optimize_for,
            backend=backend,
        )
    except (KeyError, ValueError) as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = _pipeline.compile(req, job_dir / "out")
    if not result.success:
        shutil.rmtree(job_dir, ignore_errors=True)
        err = result.error
        raise HTTPException(
            status_code=422,
            detail={"code": err.code, "message": err.message, "hints": err.hints},
        )

    project_dir = Path(result.artifacts[0]["path"])
    zip_path = job_dir / f"{project_dir.name}.zip"
    _zip_dir(project_dir, zip_path)
    _jobs[job_id] = {"zip": zip_path, "report": result.report}

    return {
        "job_id": job_id,
        "status": "done",
        "report": result.report,
        "artifact_url": f"/v1/jobs/{job_id}/artifact",
    }


@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return {"job_id": job_id, "status": "done", "report": job["report"]}


@app.get("/v1/jobs/{job_id}/artifact")
def job_artifact(job_id: str) -> FileResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return FileResponse(job["zip"], filename=Path(job["zip"]).name)


def _selection_dict(selection: SelectionReport) -> dict:
    return {
        "candidates": [
            {
                "backend": c.backend_id,
                "score": c.score,
                "op_coverage": c.op_coverage,
                "conversion_hops": c.conversion_hops,
                "estimate": c.estimate,
                "warnings": [w.message for w in c.warnings],
            }
            for c in selection.candidates
        ],
        "rejected": [
            {"backend": bid, "code": e.code, "message": e.message, "hints": e.hints}
            for bid, e in selection.rejected
        ],
    }


def _zip_dir(src: Path, dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src.parent))
