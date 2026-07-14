"""Job execution seam for the orchestrator.

The endpoints never run compile work themselves — they hand it to a
JobRunner. The MVP implementation (InProcessRunner) executes jobs on a
single worker thread and keeps state in memory; wave 2 replaces it with a
Celery-backed runner behind the same protocol, without touching the
endpoints. This module must stay importable without the [server] extra —
no FastAPI imports here.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol

#: Finished jobs (and their artifacts on disk) are dropped after this many
#: seconds so a long-running server does not fill the disk.
DEFAULT_JOB_TTL_S = int(os.environ.get("FITCHIP_JOB_TTL_S", 6 * 3600))


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobFailure(Exception):
    """An expected compile failure. `detail` is the HTTP-ready error payload
    (str or dict) — the endpoint maps it to a 422 unchanged."""

    def __init__(self, detail: str | dict):
        super().__init__(str(detail))
        self.detail = detail


@dataclass
class Job:
    id: str
    dir: Path  # uploaded model + artifacts; removed on failure or expiry
    created_at: float
    status: JobStatus = JobStatus.QUEUED
    report: dict | None = None
    error: str | dict | None = None
    zip_path: Path | None = None


#: A job's work: produce (report, artifact zip). Raise JobFailure for
#: expected compile errors; anything else is treated as an internal error.
JobWork = Callable[[], tuple[dict, Path]]


class JobRunner(Protocol):
    """The execution seam between the API and the compile work."""

    def submit(self, job_dir: Path, work: JobWork) -> Job: ...

    def get(self, job_id: str) -> Job | None: ...

    def wait(self, job_id: str) -> Job:
        """Block until the job reaches a terminal state and return it.
        Re-raises unexpected (non-JobFailure) exceptions from the work."""
        ...


@dataclass
class InProcessRunner:
    """MVP runner: one worker thread, jobs kept in memory until they expire."""

    ttl_s: int = DEFAULT_JOB_TTL_S
    _jobs: dict[str, Job] = field(default_factory=dict)
    _futures: dict[str, Future] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="fitchip-job"
        )
    )

    def submit(self, job_dir: Path, work: JobWork) -> Job:
        job = Job(id=uuid.uuid4().hex, dir=job_dir, created_at=time.time())
        with self._lock:
            self._purge_expired()
            self._jobs[job.id] = job
            self._futures[job.id] = self._executor.submit(self._run, job, work)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._purge_expired()
            return self._jobs.get(job_id)

    def wait(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs[job_id]
            future = self._futures[job_id]
        future.result()  # re-raises unexpected exceptions in the caller
        return job

    def _run(self, job: Job, work: JobWork) -> None:
        job.status = JobStatus.RUNNING
        try:
            job.report, job.zip_path = work()
            job.status = JobStatus.DONE
        except JobFailure as exc:
            job.status = JobStatus.FAILED
            job.error = exc.detail
            shutil.rmtree(job.dir, ignore_errors=True)
        except BaseException:
            job.status = JobStatus.FAILED
            job.error = "Internal error — see server logs."
            shutil.rmtree(job.dir, ignore_errors=True)
            raise

    def _purge_expired(self) -> None:
        """Drop terminal jobs older than the TTL. Queued/running jobs are
        never purged — their directories may still be in use."""
        cutoff = time.time() - self.ttl_s
        expired = [
            jid
            for jid, job in self._jobs.items()
            if job.created_at < cutoff
            and job.status in (JobStatus.DONE, JobStatus.FAILED)
        ]
        for jid in expired:
            job = self._jobs.pop(jid)
            self._futures.pop(jid, None)
            shutil.rmtree(job.dir, ignore_errors=True)
