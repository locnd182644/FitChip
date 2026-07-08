"""FitChip web GUI — Streamlit MVP (the React front-end replaces this later).

Talks directly to the core Pipeline (same process, local-first: the model
never leaves this machine). Run with:

    pip install 'fitchip[web]'
    streamlit run fitchip/web/app.py
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from fitchip.core.pipeline import Pipeline

st.set_page_config(page_title="FitChip", page_icon="✨", layout="centered")
st.title("🪄 FitChip")
st.caption("Turn a trained ML model into a ready-to-flash C/C++ firmware project.")


@st.cache_resource
def get_pipeline() -> Pipeline:
    return Pipeline()


pipeline = get_pipeline()

uploaded = st.file_uploader("Model file", type=["tflite", "onnx"])
col1, col2 = st.columns(2)
with col1:
    target_id = st.selectbox(
        "Target board", pipeline.targets.ids(), format_func=lambda t: pipeline.targets.get(t).display_name
    )
with col2:
    quantize = st.selectbox("Quantization", ["none", "int8"])

if uploaded is not None:
    workdir = Path(tempfile.mkdtemp(prefix="fitchip-web-"))
    model_path = workdir / uploaded.name
    model_path.write_bytes(uploaded.getvalue())

    req = pipeline.build_request(
        str(model_path), target_id, quantize="int8_full" if quantize == "int8" else None
    )

    # Fast lane: instant feedback before the user commits to compiling.
    try:
        meta, selection = pipeline.inspect(req)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Inspection report")
    c1, c2, c3 = st.columns(3)
    c1.metric("Size", f"{meta.file_size_bytes / 1024:.0f} KB")
    c2.metric("Operators", meta.num_ops)
    c3.metric("Quantized", "yes" if meta.is_quantized else "no")
    with st.expander("Operator breakdown"):
        st.table(
            [{"op": op, "count": n} for op, n in sorted(meta.op_counts.items(), key=lambda kv: -kv[1])]
        )

    if not selection.candidates:
        st.error("No installed backend can compile this model for this target.")
        for backend_id, err in selection.rejected:
            st.warning(f"**{backend_id}** — [{err.code}] {err.message}")
        st.stop()

    best = selection.best
    est = best.estimate
    st.success(
        f"Backend **{best.backend_id}** selected (score {best.score}) — "
        f"op coverage {best.op_coverage * 100:.0f}%"
        + (
            f", arena ≈ {est['arena_kb']} KB, flash ≈ {est['flash_kb']} KB"
            if est.get("arena_kb") is not None
            else ""
        )
    )
    for warning in best.warnings:
        st.warning(warning.message)

    if st.button("Compile", type="primary"):
        # Slow lane. Synchronous in the MVP; wave 2 submits a Celery job here
        # and polls its status instead.
        with st.spinner("Compiling…"):
            result = pipeline.compile(req, workdir / "out")
        if not result.success:
            st.error(f"[{result.error.code}] {result.error.message}")
            for hint in result.error.hints:
                st.info(hint)
        else:
            st.balloons()
            st.json(result.report)
            project_dir = Path(result.artifacts[0]["path"])
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in sorted(project_dir.rglob("*")):
                    if path.is_file():
                        zf.write(path, path.relative_to(project_dir.parent))
            st.download_button(
                "⬇ Download project ZIP",
                data=buf.getvalue(),
                file_name=f"{project_dir.name}.zip",
                mime="application/zip",
            )
