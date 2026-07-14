"""InProcessRunner tests — pure core, no [server] extra required."""

from pathlib import Path

import pytest

from fitchip.orchestrator.jobs import InProcessRunner, JobFailure, JobStatus


def _make_job_dir(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "model.tflite").write_bytes(b"x")
    return d


def _ok_work(job_dir: Path):
    zip_path = job_dir / "out.zip"
    zip_path.write_bytes(b"PK")
    return {"arena_kb": 1}, zip_path


def test_successful_job(tmp_path):
    runner = InProcessRunner()
    job_dir = _make_job_dir(tmp_path, "job1")
    job = runner.submit(job_dir, lambda: _ok_work(job_dir))
    job = runner.wait(job.id)
    assert job.status is JobStatus.DONE
    assert job.report == {"arena_kb": 1}
    assert job.zip_path.exists()
    assert runner.get(job.id) is job


def test_job_failure_is_recorded_and_dir_removed(tmp_path):
    runner = InProcessRunner()
    job_dir = _make_job_dir(tmp_path, "job1")

    def work():
        raise JobFailure({"code": "OP_UNSUPPORTED", "message": "boom", "hints": []})

    job = runner.submit(job_dir, work)
    job = runner.wait(job.id)  # expected failures do not raise
    assert job.status is JobStatus.FAILED
    assert job.error["code"] == "OP_UNSUPPORTED"
    assert not job_dir.exists()


def test_unexpected_exception_reraises_in_wait(tmp_path):
    runner = InProcessRunner()
    job_dir = _make_job_dir(tmp_path, "job1")

    def work():
        raise RuntimeError("bug")

    job = runner.submit(job_dir, work)
    with pytest.raises(RuntimeError, match="bug"):
        runner.wait(job.id)
    assert job.status is JobStatus.FAILED
    assert not job_dir.exists()


def test_ttl_purges_terminal_jobs_and_their_dirs(tmp_path):
    runner = InProcessRunner(ttl_s=0)
    job_dir = _make_job_dir(tmp_path, "job1")
    job = runner.submit(job_dir, lambda: _ok_work(job_dir))
    runner.wait(job.id)
    assert job_dir.exists()
    # Any later runner access purges expired terminal jobs.
    assert runner.get(job.id) is None
    assert not job_dir.exists()


def test_ttl_never_purges_running_jobs(tmp_path):
    import threading

    release = threading.Event()
    started = threading.Event()
    runner = InProcessRunner(ttl_s=0)
    job_dir = _make_job_dir(tmp_path, "job1")

    def work():
        started.set()
        release.wait(timeout=10)
        return _ok_work(job_dir)

    job = runner.submit(job_dir, work)
    started.wait(timeout=10)
    assert runner.get(job.id) is job  # running → survives ttl_s=0
    assert job_dir.exists()
    release.set()
    assert runner.wait(job.id).status is JobStatus.DONE
