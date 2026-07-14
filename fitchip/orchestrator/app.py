"""FitChip Orchestrator — FastAPI service in front of the core pipeline.

Two lanes, per the architecture doc:

- fast lane  (sync):  /v1/inspect — parse + validate + estimate, a few hundred
  ms, so a GUI can react before the user hits "compile".
- slow lane:          /v1/compile — runs the real compilation through a
  JobRunner (see jobs.py). MVP: InProcessRunner, one worker thread, and the
  endpoint waits so the response still carries the report inline. Wave 2
  (together with the TVM backend) swaps in a Celery-backed runner and returns
  202 + job id instead — clients already poll /v1/jobs/{id}, so they survive
  the switch unchanged.

All endpoints are plain `def`: FastAPI runs them on its threadpool, so a
minutes-long compile never blocks the event loop (or /v1/health).

Run:  uvicorn fitchip.orchestrator.app:app --reload      (pip install 'fitchip[server]')
"""

from __future__ import annotations

import dataclasses
import shutil
import tempfile
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
from fitchip.core.cal.quant import normalize_quantize
from fitchip.core.pipeline import Pipeline
from fitchip.core.selection.engine import SelectionReport
from fitchip.orchestrator.jobs import InProcessRunner, JobFailure, JobStatus

app = FastAPI(
    title="FitChip Orchestrator",
    version=fitchip.__version__,
    description="Compile trained ML models into ready-to-flash firmware projects.",
)

_pipeline = Pipeline()
_runner = InProcessRunner()


def _safe_filename(filename: str | None) -> str:
    """Client-controlled filenames may carry directory components
    ("../../etc/cron.d/x", "/etc/passwd"); keep only the final name."""
    name = Path(filename or "model").name
    return "model" if name in ("", ".", "..") else name


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
def inspect(
    model: UploadFile = File(...),
    target: str = Form(...),
    quantize: str | None = Form(None),
) -> dict:
    """Fast lane: compatibility + memory report, no compilation."""
    with tempfile.TemporaryDirectory(prefix="fitchip-inspect-") as tmp:
        model_path = Path(tmp) / _safe_filename(model.filename)
        model_path.write_bytes(model.file.read())
        try:
            req = _pipeline.build_request(
                str(model_path), target, quantize=normalize_quantize(quantize)
            )
            meta, selection = _pipeline.inspect(req)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"model": meta.to_dict(), "selection": _selection_dict(selection)}


@app.post("/v1/compile")
def compile_model(
    model: UploadFile = File(...),
    target: str = Form(...),
    quantize: str | None = Form(None),
    optimize_for: str = Form("size"),
    backend: str | None = Form(None),
) -> dict:
    """Slow lane. The MVP waits for the job so the report comes back inline;
    wave 2 returns 202 here and lets clients poll /v1/jobs/{id}."""
    job_dir = Path(tempfile.mkdtemp(prefix="fitchip-job-"))
    model_path = job_dir / _safe_filename(model.filename)
    model_path.write_bytes(model.file.read())

    try:
        req = _pipeline.build_request(
            str(model_path),
            target,
            quantize=normalize_quantize(quantize),
            optimize_for=optimize_for,
            backend=backend,
        )
    except (KeyError, ValueError) as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    def work() -> tuple[dict, Path]:
        try:
            result = _pipeline.compile(req, job_dir / "out")
        except (ValueError, FileNotFoundError) as exc:
            # e.g. weights-only checkpoints rejected by the inspector
            raise JobFailure(str(exc)) from exc
        if not result.success:
            err = result.error
            raise JobFailure(
                {"code": err.code, "message": err.message, "hints": err.hints}
            )
        project_dir = Path(result.artifacts[0]["path"])
        zip_path = job_dir / f"{project_dir.name}.zip"
        _zip_dir(project_dir, zip_path)
        return result.report, zip_path

    job = _runner.submit(job_dir, work)
    job = _runner.wait(job.id)
    if job.status is JobStatus.FAILED:
        raise HTTPException(status_code=422, detail=job.error)

    return {
        "job_id": job.id,
        "status": job.status.value,
        "report": job.report,
        "artifact_url": f"/v1/jobs/{job.id}/artifact",
    }


@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _runner.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    body: dict = {"job_id": job.id, "status": job.status.value}
    if job.status is JobStatus.DONE:
        body["report"] = job.report
        body["artifact_url"] = f"/v1/jobs/{job.id}/artifact"
    elif job.status is JobStatus.FAILED:
        body["error"] = job.error
    return body


@app.get("/v1/jobs/{job_id}/artifact")
def job_artifact(job_id: str) -> FileResponse:
    job = _runner.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    if job.status is not JobStatus.DONE or job.zip_path is None:
        raise HTTPException(
            status_code=409, detail=f"Job is {job.status.value}, no artifact available"
        )
    return FileResponse(job.zip_path, filename=job.zip_path.name)


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
