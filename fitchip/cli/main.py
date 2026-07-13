"""FitChip CLI — `fitchip compile / inspect / targets / backends / samples`.

Talks only to the core Pipeline (CAL); has no idea which backends exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

import fitchip
from fitchip.core.cal.backend import NormalizedError
from fitchip.core.cal.quant import QUANT_CHOICES, normalize_quantize
from fitchip.core.pipeline import Pipeline


def _ok(msg: str) -> None:
    click.echo(click.style("✔ ", fg="green") + msg)


def _warn(msg: str) -> None:
    click.echo(click.style("⚠ ", fg="yellow") + msg)


def _fail(err: NormalizedError) -> None:
    click.echo(click.style(f"✘ [{err.code}] ", fg="red") + err.message, err=True)
    for hint in err.hints:
        click.echo(click.style("  ↪ ", fg="cyan") + hint, err=True)


def _kb(n_bytes: int) -> str:
    return f"{n_bytes / 1024:.1f} KB" if n_bytes < 1024**2 else f"{n_bytes / 1024**2:.2f} MB"


@click.group()
@click.version_option(version=fitchip.__version__, prog_name="fitchip")
def cli() -> None:
    """Turn a trained ML model into a ready-to-flash firmware project."""


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--target", "-t", required=True, help="Target board id (see `fitchip targets`).")
@click.option("--quantize", "-q", type=click.Choice(QUANT_CHOICES), default="none",
              help="Quantization mode.")
@click.option("--calibration-data", type=click.Path(exists=True),
              help="Representative input samples (.npy file or directory) for INT8 calibration.")
@click.option("--optimize-for", type=click.Choice(["size", "speed"]), default="size")
@click.option("--backend", help="Force a specific backend instead of auto-selection.")
@click.option("--out", "-o", "out_dir", type=click.Path(file_okay=False), default="out",
              help="Output directory (default: ./out).")
def compile(model, target, quantize, calibration_data, optimize_for, backend, out_dir):
    """Compile MODEL into a buildable firmware project for --target."""
    pipeline = Pipeline()
    try:
        req = pipeline.build_request(
            model_path=model,
            target_id=target,
            quantize=normalize_quantize(quantize),
            calibration_data=calibration_data,
            optimize_for=optimize_for,
            backend=backend,
        )
    except (KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    # Fast lane first: parse + validate + estimate, so failures are cheap.
    try:
        meta, selection = pipeline.inspect(req)
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    _ok(f"Model parsed            {meta.num_ops} ops, {_kb(meta.file_size_bytes)}"
        + (" (quantized)" if meta.is_quantized else " (float32)" if meta.is_quantized is False else ""))

    if not selection.candidates:
        _fail(NormalizedError(
            code="NO_BACKEND",
            message="No installed backend can compile this model for this target:",
        ))
        for backend_id, err in selection.rejected:
            _fail(err)
        sys.exit(1)

    best = selection.best
    covered = int(best.op_coverage * meta.num_ops)
    _ok(f"Compatibility check     {covered}/{meta.num_ops} ops supported on {target} "
        f"(backend: {best.backend_id})")
    for warning in best.warnings:
        _warn(warning.message)

    est = best.estimate
    if est.get("arena_kb") is not None:
        fits = "fits" if est["arena_kb"] <= req.target.ram_kb else "EXCEEDS"
        _ok(f"Memory estimate         arena ≈ {est['arena_kb']} KB · "
            f"flash ≈ {est['flash_kb']} KB · {fits} {req.target.display_name}"
            + (" ✓" if fits == "fits" else " ✗"))

    result = pipeline.compile(req, out_dir)
    if not result.success:
        _fail(result.error)
        sys.exit(1)

    project = result.artifacts[0]["path"]
    _ok(f"Project generated       {project}/  (ESP-IDF + PlatformIO)")
    click.echo()
    click.echo(f"Next:  cd {project} && idf.py set-target "
               f"{req.target.id} && idf.py flash monitor")


@cli.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--target", "-t", help="Also check compatibility against this target.")
def inspect(model, target):
    """Compatibility & memory report for MODEL — no compilation, no hardware needed."""
    pipeline = Pipeline()
    try:
        if target is None:
            from fitchip.core.inspector import inspect_model

            meta = inspect_model(model)
            _print_meta(meta)
            return

        req = pipeline.build_request(model_path=model, target_id=target)
        meta, selection = pipeline.inspect(req)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    _print_meta(meta)

    click.echo()
    click.echo(click.style(f"Compatibility with {target}:", bold=True))
    for cand in selection.candidates:
        est = cand.estimate
        arena = f"{est['arena_kb']} KB" if est.get("arena_kb") is not None else "n/a (post-conversion)"
        click.echo(
            f"  {click.style('●', fg='green')} {cand.backend_id}: score {cand.score}, "
            f"op coverage {cand.op_coverage * 100:.0f}%, arena ≈ {arena}, "
            f"flash ≈ {est.get('flash_kb', '?')} KB"
        )
        for warning in cand.warnings:
            _warn(f"    {warning.message}")
    for backend_id, err in selection.rejected:
        click.echo(f"  {click.style('●', fg='red')} {backend_id}: [{err.code}] {err.message}")
        for hint in err.hints:
            click.echo(click.style("      ↪ ", fg="cyan") + hint)


def _print_meta(meta) -> None:
    _ok(f"Format                  {meta.format}")
    _ok(f"Size                    {_kb(meta.file_size_bytes)}")
    _ok(f"Operators               {meta.num_ops} total, {len(meta.op_counts)} unique")
    for op, count in sorted(meta.op_counts.items(), key=lambda kv: -kv[1]):
        click.echo(f"    {op:<32} × {count}")
    if meta.custom_ops:
        _warn(f"Custom ops: {', '.join(meta.custom_ops)}")
    for io_kind, tensors in (("Inputs", meta.inputs), ("Outputs", meta.outputs)):
        desc = "; ".join(f"{t['name']} {t['shape']} {t['dtype']}" for t in tensors)
        _ok(f"{io_kind:<23} {desc}")
    if meta.is_quantized is not None:
        _ok(f"Quantized               {'yes' if meta.is_quantized else 'no (float32)'}")
    for warning in meta.warnings:
        _warn(warning)


@cli.command()
def targets():
    """List supported hardware targets."""
    from fitchip.core.targets import TargetRegistry

    for t in TargetRegistry().all():
        accel = f" · {', '.join(t.accelerators)}" if t.accelerators else ""
        click.echo(
            f"{t.id:<12} {t.display_name:<12} RAM {t.ram_kb} KB · "
            f"flash {t.flash_kb} KB · {t.isa}{accel}"
        )


@cli.command()
def backends():
    """List installed compiler backends."""
    from fitchip.core.cal.registry import BackendRegistry

    for backend in BackendRegistry().all():
        caps = backend.capabilities()
        click.echo(
            f"{caps['id']:<8} {caps['display_name']} — in: {', '.join(caps['input_formats'])}"
            f" · quant: {', '.join(str(q) for q in caps['quantization'])}"
        )


@cli.group()
def samples():
    """Verified sample models."""


@samples.command("list")
def samples_list():
    from fitchip import samples as samples_mod

    for name, entry in sorted(samples_mod.registry().items()):
        click.echo(f"{name:<16} {entry['description']}")


@samples.command("pull")
@click.argument("name")
@click.option("--out", "-o", "dest", type=click.Path(file_okay=False), default=".")
def samples_pull(name, dest):
    """Download a sample model into the current directory."""
    from fitchip import samples as samples_mod

    Path(dest).mkdir(parents=True, exist_ok=True)
    try:
        path = samples_mod.pull(name, dest)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    _ok(f"Downloaded {path}")


if __name__ == "__main__":
    cli()
