"""
app.py — Adaptive Placement Intelligence
========================================
The whole product surface. Two things a user does: train a model on their own
spreadsheet, then predict with it until they choose to retrain.

    Sidebar   workspace, provider status, active model
    Tab 1     Train    upload -> plan -> train -> save
    Tab 2     Predict  generated form + batch scoring
    Tab 3     History  what this workspace has predicted, and drift
    Tab 4     Model    the model card: plan, features, scores, provenance

Streamlit reruns this entire file on every widget interaction, so nothing here
holds state in a local variable. What survives a rerun is either in
st.session_state (which workspace is open, what was last trained) or on disk
(workspaces, models, history). Anything expensive sits behind a cache keyed on
identifiers rather than objects.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from placement_ai.config import SAMPLE_DATASET_PATH, WORKSPACE_ROOT
from placement_ai.inference.drift import check_drift
from placement_ai.inference.explain import (
    attribute,
    cohort_summary,
    improvement_levers,
    what_if_curve,
)
from placement_ai.inference.predictor import WorkspacePredictor
from placement_ai.llm.registry import (
    dotenv_status,
    get_provider,
    provider_key_names,
    provider_status,
)
from placement_ai.planner.narrator import provenance_summary, write_prediction_advice
from placement_ai.plans import TrainingPlan
from placement_ai.profiling import canonicalize_columns, classify_target_labels, profile_dataframe
from placement_ai.registry.history import HistoryEntry, PredictionHistory
from placement_ai.registry.model_store import ModelBundle, ModelStore, ModelStoreError
from placement_ai.registry.workspace import Workspace, WorkspaceError, WorkspaceStore
from placement_ai.training.runner import STAGE_TITLES, TrainingError, TrainingRunner
from placement_ai.utils import human_duration

# =============================================================================
# 1. PAGE SETUP & SHARED STYLING
# =============================================================================
st.set_page_config(
    page_title="Adaptive Placement Intelligence",
    page_icon=":material/network_intelligence:",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHART_HEIGHT = 320


def chart_layout(height: int = CHART_HEIGHT, title: str | None = None) -> dict[str, Any]:
    """One Plotly layout for every chart, matching .streamlit/config.toml."""
    layout: dict[str, Any] = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "color": "#CBD5E1", "size": 12},
        "margin": {"l": 24, "r": 24, "t": 40 if title else 16, "b": 24},
        "height": height,
        "xaxis": {"gridcolor": "#334155", "zerolinecolor": "#334155"},
        "yaxis": {"gridcolor": "#334155", "zerolinecolor": "#334155"},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    }
    if title:
        layout["title"] = {"text": title, "font": {"size": 15, "color": "#F8FAFC"}}
    return layout


POSITIVE = "#34D399"
NEGATIVE = "#F87171"
NEUTRAL = "#60A5FA"
MUTED = "#94A3B8"


# =============================================================================
# 2. CACHED RESOURCES
# =============================================================================


@st.cache_resource(show_spinner=False)
def get_store() -> WorkspaceStore:
    return WorkspaceStore(WORKSPACE_ROOT)


@st.cache_resource(show_spinner="Loading the model...")
def load_predictor(workspace_id: str, version: str, checksum: str) -> WorkspacePredictor:
    """Deserialise one model, once per process.

    Keyed on the checksum as well as the version so that a retrain writing over
    the same version — or a corrupted file — produces a cache miss rather than
    silently serving the previous object.
    """
    workspace = get_store().get(workspace_id)
    if workspace is None:
        raise ModelStoreError("That workspace is no longer available.")
    bundle = ModelStore(workspace).get(version)
    if bundle is None:
        raise ModelStoreError(f"Model {version} is no longer available.")
    return WorkspacePredictor(bundle)


def predictor_for(workspace: Workspace, bundle: ModelBundle) -> WorkspacePredictor:
    checksum = (bundle.manifest.get("artifacts", {}).get("pipeline.joblib", {}) or {}).get(
        "sha256", bundle.version
    )
    return load_predictor(workspace.id, bundle.version, checksum)


@st.cache_data(show_spinner=False)
def read_upload(data: bytes, name: str) -> pd.DataFrame:
    """Parse an uploaded CSV or Excel file, cached on its bytes."""
    buffer = io.BytesIO(data)
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)


@st.cache_data(show_spinner=False)
def profile_upload(data: bytes, name: str) -> tuple[pd.DataFrame, list[str], int]:
    frame, _ = canonicalize_columns(read_upload(data, name))
    profile = profile_dataframe(frame)
    return frame, profile.target_candidates, profile.n_columns


# =============================================================================
# 3. SESSION STATE
# =============================================================================
st.session_state.setdefault("workspace_id", None)
st.session_state.setdefault("workspace_code", "")
st.session_state.setdefault("last_trained", None)
st.session_state.setdefault("new_workspace_code", None)
st.session_state.setdefault("prediction", None)
st.session_state.setdefault("use_sample", False)


def current_workspace() -> Workspace | None:
    workspace_id = st.session_state.get("workspace_id")
    return get_store().get(workspace_id) if workspace_id else None


def close_workspace() -> None:
    for key in ("workspace_id", "workspace_code", "last_trained", "prediction"):
        st.session_state[key] = None if key != "workspace_code" else ""


# =============================================================================
# 4. SIDEBAR — WORKSPACE & PROVIDER
# =============================================================================
def render_sidebar() -> Workspace | None:
    store = get_store()
    workspace = current_workspace()

    with st.sidebar:
        st.markdown("### :material/network_intelligence: Adaptive Placement Intelligence")
        st.caption("Train a model on your own data, then predict with it.")

        if workspace is None:
            render_workspace_gate(store)
        else:
            render_workspace_panel(workspace)

        st.divider()
        render_provider_panel()

    return current_workspace()


def render_workspace_gate(store: WorkspaceStore) -> None:
    """Create or open a workspace. Nothing else in the app works until one is."""
    existing = store.list()

    open_tab, create_tab = st.tabs(["Open", "Create"])

    with open_tab:
        if not existing:
            st.info("No workspaces yet. Create one to get started.")
        else:
            choice = st.selectbox(
                "Workspace",
                options=[w.id for w in existing],
                format_func=lambda wid: next(w.name for w in existing if w.id == wid),
            )
            code = st.text_input("Access code", type="password", key="open_code")
            if st.button("Open", type="primary", width="stretch"):
                try:
                    opened = store.open(choice, code)
                except WorkspaceError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["workspace_id"] = opened.id
                    st.session_state["workspace_code"] = code
                    st.rerun()

    with create_tab:
        name = st.text_input("Organisation name", placeholder="St. Xavier College")
        description = st.text_input("Description", placeholder="Placement cell 2026")
        if st.button("Create workspace", type="primary", width="stretch"):
            try:
                created, code = store.create(name, description)
            except WorkspaceError as exc:
                st.error(str(exc))
            else:
                st.session_state["workspace_id"] = created.id
                st.session_state["workspace_code"] = code
                st.session_state["new_workspace_code"] = code
                st.rerun()

    if st.session_state.get("new_workspace_code"):
        st.success("Workspace created.")
        st.code(st.session_state["new_workspace_code"], language=None)
        st.caption("Save this access code — it is shown once and cannot be recovered.")


def render_workspace_panel(workspace: Workspace) -> None:
    store = ModelStore(workspace)
    bundles = store.list()
    active = store.active()

    st.success(f"**{workspace.name}**", icon=":material/domain:")
    if workspace.description:
        st.caption(workspace.description)

    if st.session_state.get("new_workspace_code"):
        st.warning("Access code — shown once:", icon=":material/key:")
        st.code(st.session_state["new_workspace_code"], language=None)
        if st.button("I have saved it", width="stretch"):
            st.session_state["new_workspace_code"] = None
            st.rerun()

    st.markdown("##### :material/model_training: Active model")
    if not bundles:
        st.caption("No model yet. Train one in the Train tab.")
    else:
        selected = st.selectbox(
            "Version",
            options=[b.version for b in bundles],
            index=next((i for i, b in enumerate(bundles) if active and b.version == active.version), 0),
            format_func=lambda v: _bundle_caption(next(b for b in bundles if b.version == v)),
            label_visibility="collapsed",
        )
        if active is None or selected != active.version:
            if st.button("Use this version", type="primary", width="stretch"):
                store.set_active(selected)
                st.rerun()
        else:
            score = active.headline_score
            metric = active.primary_metric.replace("_", " ")
            st.caption(
                f"{active.label} · {metric} {score:.3f}"
                if score is not None
                else active.label
            )

    st.divider()
    if st.button("Close workspace", width="stretch"):
        close_workspace()
        st.rerun()


def _bundle_caption(bundle: ModelBundle) -> str:
    score = bundle.headline_score
    stamp = bundle.created_at.replace("T", " ").replace("Z", "")
    return f"{stamp} · {bundle.label}" + (f" · {score:.3f}" if score is not None else "")


PROVIDER_LABELS = {"gemini": "Gemini", "grok": "Grok (xAI)", "openrouter": "OpenRouter"}


def render_provider_panel() -> None:
    st.markdown("##### :material/smart_toy: AI planner")

    # A .env that cannot be read is the one failure that looks identical to
    # having no key at all, so it is reported before anything else.
    problem = dotenv_status()
    if problem:
        st.error(problem, icon=":material/report:")

    status = provider_status()
    provider = get_provider()

    if provider is None:
        st.warning("Running on built-in rules", icon=":material/rule:")
        st.caption(
            "No API key found. Training still works end to end — the planner uses "
            "its rule-based fallback instead of a language model."
        )
    else:
        st.info(f"{provider.name} · {provider.model}", icon=":material/bolt:")
        st.caption("The planner reads your columns and designs the pipeline.")

    with st.expander("Providers"):
        st.caption(
            "Set a key as an environment variable, in a `.env` file at the repo "
            "root, or in `.streamlit/secrets.toml`, then restart the app. "
            "`LLM_PROVIDER` picks between them; `off` forces the rule-based planner."
        )
        for kind, key_names in provider_key_names().items():
            found = status.get(kind, False)
            icon = ":material/check_circle:" if found else ":material/circle:"
            names = " or ".join(f"`{name}`" for name in key_names)
            st.markdown(f"{icon} **{PROVIDER_LABELS.get(kind, kind)}** — {names}")
        st.caption(
            "A key being present does not mean it works — an unfunded account or a "
            "retired model still falls back. The model card records what actually "
            "planned each run."
        )


# =============================================================================
# 5. TAB 1: TRAIN
# =============================================================================
def render_train_tab(workspace: Workspace) -> None:
    st.subheader(":material/upload_file: Train a model on your data")
    st.caption(
        "Upload a spreadsheet where one column records the outcome you want to predict. "
        "Everything else — cleaning, feature design, model choice — is planned for you."
    )

    # The result of a finished run lives on disk, so it is re-rendered from the
    # saved bundle rather than held in memory. Without this, the summary would
    # disappear the moment the user touched any other widget.
    last = st.session_state.get("last_trained")
    if last:
        bundle = ModelStore(workspace).get(last)
        if bundle is not None:
            with st.container(border=True):
                render_training_result(bundle)
                if st.button("Train another model", icon=":material/refresh:"):
                    st.session_state["last_trained"] = None
                    st.session_state["use_sample"] = False
                    st.rerun()
            st.divider()

    upload = st.file_uploader(
        "Training data", type=["csv", "xlsx", "xls"], label_visibility="collapsed"
    )

    # A button press only survives one rerun, so the choice is parked in session
    # state — otherwise clicking "use sample" reruns the script and immediately
    # forgets it, leaving the tab looking like nothing happened.
    if upload is not None:
        st.session_state["use_sample"] = False
    if upload is None:
        left, right = st.columns([3, 1])
        left.info(
            "No file yet. You can try the bundled sample — 10,000 student records "
            "with placement outcomes.",
            icon=":material/lightbulb:",
        )
        if right.button(
            "Use sample data", width="stretch", disabled=not SAMPLE_DATASET_PATH.exists()
        ):
            st.session_state["use_sample"] = True
            st.rerun()
        if not st.session_state.get("use_sample"):
            return

    if upload is None:
        data = SAMPLE_DATASET_PATH.read_bytes()
        filename = SAMPLE_DATASET_PATH.name
    else:
        data = upload.getvalue()
        filename = upload.name

    try:
        frame, candidates, n_columns = profile_upload(data, filename)
    except Exception as exc:
        st.error(f"That file could not be read: {type(exc).__name__}: {exc}")
        return

    st.success(
        f"**{filename}** — {len(frame):,} rows x {n_columns} columns",
        icon=":material/table_view:",
    )
    with st.expander("Preview the first rows"):
        st.dataframe(frame.head(20), width="stretch", height=280)

    # ── target selection ─────────────────────────────────────────────────
    st.markdown("##### :material/target: What should the model predict?")
    options = candidates + [c for c in frame.columns if c not in candidates]
    target = st.selectbox(
        "Outcome column",
        options=options,
        help="The column holding the answer you want predicted for new records.",
    )

    info = classify_target_labels(frame[target])
    if info["task_type"] == "regression":
        st.error(
            f"**{target}** holds {info['n_classes']:,} distinct values, which is a "
            "quantity to estimate rather than an outcome to classify. This build "
            "predicts categories — pick a column with a small set of repeated answers."
        )
        return

    levels = info["levels"]
    columns = st.columns(min(len(levels), 4) or 1)
    for column, level in zip(columns, levels[:4], strict=False):
        share = level["count"] / max(len(frame), 1) * 100
        column.metric(str(level["value"]), f"{level['count']:,}", f"{share:.1f}% of rows")
    if info["positive_class"]:
        st.caption(
            f"**{info['positive_class']}** will be treated as the positive outcome — "
            "the one probabilities refer to."
        )

    objective = st.text_area(
        "What are you trying to achieve? (optional)",
        placeholder="Spot final-year students who need extra interview preparation before campus drives begin.",
        help="Given to the planner as context. It shapes which columns are kept and which features get built.",
        height=80,
    )

    if not st.button("Start training", type="primary", icon=":material/play_arrow:"):
        return

    run_training(workspace, frame, target, objective, filename)


def run_training(
    workspace: Workspace,
    frame: pd.DataFrame,
    target: str,
    objective: str,
    filename: str,
) -> None:
    """Execute a training run, streaming stage progress into the page."""
    provider = get_provider()
    lines: list[str] = []

    with st.status("Training in progress...", expanded=True) as status:
        progress_bar = st.progress(0.0)
        log = st.empty()

        def on_progress(event) -> None:
            title = STAGE_TITLES.get(event.stage, event.stage)
            if event.status in {"ok", "fallback"}:
                marker = "[ok]" if event.status == "ok" else "[rules]"
                lines.append(f"{marker:<8} {title} — {event.detail}")
            elif event.status == "info":
                lines.append(f"{'':<8} {event.detail}")
            else:
                lines.append(f"{'...':<8} {title} — {event.detail}")
            progress_bar.progress(event.fraction)
            log.code("\n".join(lines[-16:]), language=None)

        try:
            outcome = TrainingRunner(provider=provider, progress=on_progress).run(
                frame, target_column=target, objective=objective
            )
        except TrainingError as exc:
            status.update(label="Training stopped", state="error")
            st.error(str(exc))
            return
        except Exception as exc:
            status.update(label="Training failed", state="error")
            st.error(f"Unexpected failure: {type(exc).__name__}: {exc}")
            st.exception(exc)
            return

        try:
            bundle = ModelStore(workspace).save(
                outcome, dataset_name=filename, notes=objective, make_active=True
            )
        except Exception as exc:
            status.update(label="Model could not be saved", state="error")
            st.error(f"Training finished but saving failed: {type(exc).__name__}: {exc}")
            return

        progress_bar.progress(1.0)
        status.update(
            label=f"Trained and saved in {human_duration(outcome.duration_seconds)}",
            state="complete",
            expanded=False,
        )

    st.session_state["last_trained"] = bundle.version
    st.session_state["prediction"] = None
    render_training_result(bundle)


def render_training_result(bundle: ModelBundle) -> None:
    narrative = bundle.manifest.get("narrative", {})
    metrics = bundle.manifest.get("metrics", {})

    st.success(narrative.get("headline", "Training complete."), icon=":material/verified:")
    st.markdown(narrative.get("summary", ""))

    render_metric_row(bundle, metrics)

    left, right = st.columns(2)
    with left:
        if narrative.get("strengths"):
            st.markdown("**What this model does well**")
            for item in narrative["strengths"]:
                st.markdown(f"- {item}")
    with right:
        if narrative.get("cautions"):
            st.markdown("**Where to be careful**")
            for item in narrative["cautions"]:
                st.markdown(f"- {item}")

    if narrative.get("next_steps"):
        with st.expander("Suggested next steps"):
            for item in narrative["next_steps"]:
                st.markdown(f"- {item}")

    warnings = bundle.manifest.get("warnings") or []
    if warnings:
        with st.expander(f"{len(warnings)} note(s) from the run"):
            for warning in warnings:
                st.markdown(f"- {warning}")

    st.info(
        f"Saved as version **{bundle.version}** and now active. "
        "Head to the Predict tab to use it.",
        icon=":material/save:",
    )


def render_metric_row(bundle: ModelBundle, metrics: dict[str, Any]) -> None:
    primary = bundle.primary_metric
    tiles = [
        (primary.replace("_", " ").title(), metrics.get(primary)),
        ("Accuracy", metrics.get("accuracy")),
        ("Precision", metrics.get("precision")),
        ("Recall", metrics.get("recall")),
        ("F1", metrics.get("f1")),
    ]
    columns = st.columns(len(tiles))
    for column, (label, value) in zip(columns, tiles, strict=True):
        column.metric(label, f"{value:.3f}" if isinstance(value, (int, float)) else "—")


# =============================================================================
# 6. TAB 2: PREDICT
# =============================================================================
def render_predict_tab(workspace: Workspace, bundle: ModelBundle | None) -> None:
    if bundle is None:
        st.info("Train a model first — the Train tab walks you through it.", icon=":material/school:")
        return

    try:
        predictor = predictor_for(workspace, bundle)
    except ModelStoreError as exc:
        st.error(str(exc))
        return

    st.caption(
        f"Predicting **{bundle.target_column}** with {bundle.label} "
        f"(version {bundle.version})."
    )

    single_tab, batch_tab = st.tabs(
        [":material/person: One record", ":material/groups: A whole file"]
    )
    with single_tab:
        render_single_prediction(workspace, bundle, predictor)
    with batch_tab:
        render_batch_prediction(workspace, bundle, predictor)


def render_single_prediction(
    workspace: Workspace, bundle: ModelBundle, predictor: WorkspacePredictor
) -> None:
    st.markdown("##### Enter the details")
    st.caption("Every field, its range, and its typical value come from the data this model was trained on.")

    with st.form("single_prediction"):
        values = render_input_form(predictor)
        reference = st.text_input(
            "Reference (optional)", placeholder="Student roll number, so you can find this later"
        )
        submitted = st.form_submit_button("Predict", type="primary", icon=":material/bolt:")

    if submitted:
        record = predictor.build_record(values)
        prediction = predictor.predict_one(record)
        history = PredictionHistory(workspace.history_path)
        history.log(
            HistoryEntry(
                model_version=bundle.version,
                target_column=bundle.target_column,
                inputs=record,
                predicted_label=prediction.label,
                probability=prediction.probability,
                probabilities=prediction.probabilities,
                source="single",
                reference=reference or None,
            )
        )
        st.session_state["prediction"] = {"record": record, "reference": reference}

    stored = st.session_state.get("prediction")
    if stored:
        render_prediction_result(predictor, bundle, stored["record"])


def render_input_form(predictor: WorkspacePredictor) -> dict[str, Any]:
    """Build the form from the bundle's own description of its inputs.

    Nothing about these fields is hardcoded — a model trained on a completely
    different spreadsheet renders a completely different form here.
    """
    values: dict[str, Any] = {}
    schema = predictor.input_schema
    columns = st.columns(3)

    for index, spec in enumerate(schema):
        name = str(spec["name"])
        label = str(spec.get("label", name))
        target = columns[index % 3]
        help_text = spec.get("description") or None

        if spec.get("kind") == "categorical":
            levels = [str(level) for level in (spec.get("levels") or [])]
            if not levels:
                continue
            default = str(spec.get("default", levels[0]))
            values[name] = target.selectbox(
                label,
                options=levels,
                index=levels.index(default) if default in levels else 0,
                key=f"in_{name}",
                help=help_text,
            )
            continue

        low, high = float(spec.get("min", 0.0)), float(spec.get("max", 1.0))
        default = float(spec.get("default", low))
        if high <= low:
            # A constant column has nothing to vary; show it, do not offer a slider.
            target.number_input(label, value=default, disabled=True, key=f"in_{name}", help=help_text)
            values[name] = default
            continue

        if spec.get("integer"):
            values[name] = target.slider(
                label,
                min_value=int(low),
                max_value=int(high),
                value=int(round(default)),
                step=1,
                key=f"in_{name}",
                help=help_text,
            )
        else:
            values[name] = target.slider(
                label,
                min_value=low,
                max_value=high,
                value=min(max(default, low), high),
                step=float(spec.get("step", (high - low) / 100)),
                key=f"in_{name}",
                help=help_text,
            )
    return values


def render_prediction_result(
    predictor: WorkspacePredictor, bundle: ModelBundle, record: dict[str, Any]
) -> None:
    prediction = predictor.predict_one(record)
    positive = predictor.positive_class

    st.divider()
    headline, gauge = st.columns([1, 2])

    with headline:
        is_positive = positive is not None and prediction.label == positive
        st.metric(bundle.target_column.replace("_", " ").title(), prediction.label)
        st.metric(
            f"Confidence in {positive}" if positive else "Confidence",
            f"{prediction.probability * 100:.1f}%",
        )
        badge = {"High confidence": "green", "Moderate confidence": "orange", "Borderline": "grey"}
        st.badge(prediction.confidence_band, color=badge.get(prediction.confidence_band, "grey"))
        if not is_positive and positive:
            st.caption(f"Below the {prediction.threshold:.2f} decision threshold for {positive}.")

    with gauge:
        st.plotly_chart(
            probability_gauge(prediction.probability, prediction.threshold, positive),
            width="stretch",
        )

    drivers = attribute(predictor, record, top_n=8)
    if drivers:
        st.markdown("##### :material/insights: What moved this prediction")
        st.plotly_chart(driver_chart(drivers), width="stretch")
        st.caption(
            "Each bar is the change in probability caused by this record's value versus a "
            "typical one. Measured one field at a time, so the bars rank influence rather "
            "than adding up to the total."
        )

    levers = improvement_levers(predictor, record)
    if levers:
        st.markdown("##### :material/trending_up: Smallest changes that would flip this")
        for lever in levers:
            st.markdown(
                f"- **{lever['label']}**: {lever['from']} → {lever['to']} "
                f"(probability +{lever['gain']:.3f})"
            )
    elif positive and prediction.label != positive:
        st.caption(
            "No single field, changed on its own within the range seen in training, "
            "is enough to flip this prediction."
        )

    with st.expander(":material/auto_awesome: Written explanation"):
        if st.button("Write it", key="advice_button"):
            with st.spinner("Writing..."):
                advice = write_prediction_advice(
                    get_provider(),
                    target_column=bundle.target_column,
                    predicted_label=prediction.label,
                    probability=prediction.probability,
                    drivers=drivers,
                    inputs=record,
                    positive_class=positive,
                )
            st.markdown(f"**{advice.get('headline', '')}**")
            st.markdown(advice.get("reading", ""))
            for action in advice.get("actions", []):
                st.markdown(f"- {action}")
            if advice.get("source") == "template":
                st.caption("Written from the numbers directly — no language model was available.")

    render_what_if(predictor, record)


def render_what_if(predictor: WorkspacePredictor, record: dict[str, Any]) -> None:
    with st.expander(":material/tune: Explore one field"):
        names = [str(spec["name"]) for spec in predictor.input_schema]
        if not names:
            return
        labels = {str(s["name"]): str(s.get("label", s["name"])) for s in predictor.input_schema}
        column = st.selectbox(
            "Field", options=names, format_func=lambda n: labels.get(n, n), key="whatif_field"
        )
        curve = what_if_curve(predictor, record, column)
        if curve.empty:
            st.caption("This field cannot be varied.")
            return

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=curve["value"],
                y=curve["probability"],
                mode="lines+markers",
                line={"color": NEUTRAL, "width": 3},
                name="Probability",
            )
        )
        figure.add_hline(
            y=predictor.threshold,
            line_dash="dash",
            line_color=MUTED,
            annotation_text="decision threshold",
        )
        current = record.get(column)
        if isinstance(current, (int, float)):
            figure.add_vline(x=current, line_dash="dot", line_color=POSITIVE, annotation_text="now")
        figure.update_layout(**chart_layout(300, f"Probability as {labels.get(column, column)} changes"))
        st.plotly_chart(figure, width="stretch")
        st.caption("Only the range seen during training is shown — beyond it the model has no evidence.")


def render_batch_prediction(
    workspace: Workspace, bundle: ModelBundle, predictor: WorkspacePredictor
) -> None:
    st.markdown("##### Score a whole file")
    st.caption("Upload records without the outcome column. Everything else can stay as it is.")

    upload = st.file_uploader(
        "Records to score", type=["csv", "xlsx", "xls"], key="batch_upload",
        label_visibility="collapsed",
    )
    if upload is None:
        return

    try:
        raw = read_upload(upload.getvalue(), upload.name)
    except Exception as exc:
        st.error(f"That file could not be read: {type(exc).__name__}: {exc}")
        return

    frame, check = predictor.check_frame(raw)
    if not check.usable:
        st.error(
            "None of the columns in this file match what the model expects. "
            f"It needs: {', '.join(check.expected[:12])}"
            + ("…" if len(check.expected) > 12 else "")
        )
        return

    if check.missing:
        st.warning(
            f"{len(check.missing)} expected column(s) are absent and will be filled with "
            f"typical values: {', '.join(check.missing)}. Predictions will be less reliable "
            "for records that depend on them.",
            icon=":material/warning:",
        )

    if not st.button("Score these records", type="primary", icon=":material/bolt:"):
        return

    with st.spinner(f"Scoring {len(frame):,} records..."):
        scored = predictor.predict_frame(frame)
        history = PredictionHistory(workspace.history_path)
        batch_id = f"batch-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
        # to_dict("records") rather than iterrows(): the latter builds a Series
        # per row and turns a 10k-record file into a visible wait.
        supplied = [c for c in predictor.expected_columns if c in scored.columns]
        written = history.log_many(
            [
                HistoryEntry(
                    model_version=bundle.version,
                    target_column=bundle.target_column,
                    inputs=inputs,
                    predicted_label=str(label),
                    probability=float(probability),
                    source="batch",
                    batch_id=batch_id,
                )
                for inputs, label, probability in zip(  # noqa: B905
                    scored[supplied].to_dict("records"),
                    scored["prediction"].tolist(),
                    scored["probability"].tolist(),
                )
            ]
        )

    summary = cohort_summary(scored, predictor)
    tiles = st.columns(4)
    tiles[0].metric("Records scored", f"{summary['total']:,}")
    if summary.get("positive_rate") is not None:
        tiles[1].metric(f"Predicted {predictor.positive_class}", f"{summary['positive_rate'] * 100:.1f}%")
    tiles[2].metric("Mean probability", f"{summary['mean_probability']:.3f}")
    tiles[3].metric("Borderline cases", f"{summary['borderline']:,}")

    if written is None:
        st.caption("Scored, but these predictions could not be written to the history log.")

    left, right = st.columns(2)

    counts = pd.Series(summary["by_label"]).reset_index()
    counts.columns = ["outcome", "records"]
    figure = px.bar(counts, x="outcome", y="records", color="outcome",
                    color_discrete_sequence=[POSITIVE, NEGATIVE, NEUTRAL])
    figure.update_layout(**chart_layout(300, "Predicted outcomes"), showlegend=False)
    left.plotly_chart(figure, width="stretch")

    figure = px.histogram(scored, x="probability", nbins=40, color_discrete_sequence=[NEUTRAL])
    figure.add_vline(x=predictor.threshold, line_dash="dash", line_color=MUTED)
    figure.update_layout(**chart_layout(300, "Probability distribution"))
    right.plotly_chart(figure, width="stretch")

    st.dataframe(scored, width="stretch", height=380)
    st.download_button(
        "Download scored records",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name=f"scored_{upload.name.rsplit('.', 1)[0]}.csv",
        mime="text/csv",
        icon=":material/download:",
    )


# =============================================================================
# 7. TAB 3: HISTORY & DRIFT
# =============================================================================
def render_history_tab(workspace: Workspace, bundle: ModelBundle | None) -> None:
    history = PredictionHistory(workspace.history_path)
    summary = history.summary()

    if not summary["total"]:
        st.info("No predictions recorded yet.", icon=":material/history:")
        return

    tiles = st.columns(4)
    tiles[0].metric("Predictions made", f"{summary['total']:,}")
    tiles[1].metric("Outcomes confirmed", f"{summary['labelled']:,}")
    tiles[2].metric("Model versions used", f"{len(summary['by_version']):,}")
    tiles[3].metric("Most recent", (summary["last"] or "—").replace("T", " ").replace("Z", ""))

    frame = history.expanded(limit=5000)
    if frame.empty:
        st.caption("The history could not be read.")
        return

    left, right = st.columns(2)
    counts = pd.Series(summary["by_label"]).reset_index()
    counts.columns = ["outcome", "records"]
    figure = px.pie(counts, names="outcome", values="records", hole=0.55,
                    color_discrete_sequence=[POSITIVE, NEGATIVE, NEUTRAL])
    figure.update_layout(**chart_layout(300, "Everything predicted so far"))
    left.plotly_chart(figure, width="stretch")

    if "probability" in frame.columns:
        figure = px.histogram(frame, x="probability", nbins=40, color_discrete_sequence=[NEUTRAL])
        figure.update_layout(**chart_layout(300, "Confidence distribution"))
        right.plotly_chart(figure, width="stretch")

    render_drift_panel(bundle, frame)

    st.markdown("##### :material/table: Recent predictions")
    display = frame.drop(columns=[c for c in ("id", "batch_id") if c in frame.columns])
    st.dataframe(display.head(300), width="stretch", height=360)
    st.download_button(
        "Download full history",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{workspace.id}_prediction_history.csv",
        mime="text/csv",
        icon=":material/download:",
    )

    render_outcome_recorder(workspace, history, bundle)


def render_drift_panel(bundle: ModelBundle | None, frame: pd.DataFrame) -> None:
    if bundle is None:
        return
    st.markdown("##### :material/monitoring: Has your population shifted?")
    report = check_drift(bundle.drift_baseline, frame, bundle.input_schema)

    banner = {
        "ok": st.success,
        "watch": st.warning,
        "alert": st.error,
        "insufficient_data": st.info,
        "unavailable": st.info,
    }[report.status]
    banner(report.message, icon=":material/monitoring:")

    if not report.columns:
        return

    top = report.columns[:12]
    figure = px.bar(
        pd.DataFrame({"column": [c.label for c in top], "psi": [c.psi for c in top]}),
        x="psi", y="column", orientation="h",
        color=[c.status for c in top],
        color_discrete_map={"ok": POSITIVE, "watch": "#FBBF24", "alert": NEGATIVE},
    )
    figure.add_vline(x=0.10, line_dash="dot", line_color=MUTED)
    figure.add_vline(x=0.25, line_dash="dash", line_color=NEGATIVE)
    figure.update_layout(**chart_layout(340, "Population stability index by column"), showlegend=False)
    figure.update_yaxes(autorange="reversed")
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Compares the records you have been scoring against the data this model trained on. "
        "Below 0.10 is stable, above 0.25 means retraining is due."
    )


def render_outcome_recorder(
    workspace: Workspace, history: PredictionHistory, bundle: ModelBundle | None
) -> None:
    if bundle is None:
        return
    with st.expander(":material/fact_check: Record what actually happened"):
        st.caption(
            "Confirmed outcomes become training data. Export them once you have enough "
            "and include them in your next training file."
        )
        recent = history.recent(limit=100)
        if recent.empty:
            return

        pending = recent[recent["actual_label"].isna()]
        if pending.empty:
            st.caption("Every recent prediction already has a confirmed outcome.")
        else:
            options = pending["id"].tolist()
            chosen = st.selectbox(
                "Prediction",
                options=options,
                format_func=lambda rid: _history_caption(pending, rid),
            )
            actual = st.selectbox("What actually happened", options=bundle.class_labels)
            if st.button("Save outcome"):
                if history.record_actual(int(chosen), actual):
                    st.success("Recorded.")
                    st.rerun()
                else:
                    st.error(history.last_error or "Could not save that outcome.")

        labelled = history.labelled()
        if not labelled.empty:
            st.download_button(
                f"Download {len(labelled):,} confirmed outcome(s) as training data",
                data=labelled.to_csv(index=False).encode("utf-8"),
                file_name=f"{workspace.id}_confirmed_outcomes.csv",
                mime="text/csv",
                icon=":material/download:",
            )


def _history_caption(frame: pd.DataFrame, row_id: int) -> str:
    row = frame[frame["id"] == row_id].iloc[0]
    stamp = str(row["created_at"]).replace("T", " ").replace("Z", "")
    reference = row.get("reference") or f"#{row_id}"
    return f"{reference} · {stamp} · predicted {row['predicted_label']}"


# =============================================================================
# 8. TAB 4: MODEL CARD
# =============================================================================
def render_model_tab(workspace: Workspace, bundle: ModelBundle | None) -> None:
    store = ModelStore(workspace)
    bundles = store.list()
    if not bundles:
        st.info("No models yet.", icon=":material/inventory_2:")
        return

    versions = [b.version for b in bundles]
    active_index = versions.index(bundle.version) if bundle and bundle.version in versions else 0
    chosen = st.selectbox(
        "Model version",
        options=versions,
        index=active_index,
        format_func=lambda v: _bundle_caption(next(b for b in bundles if b.version == v)),
    )
    selected = store.get(chosen)
    if selected is None:
        return

    manifest = selected.manifest
    metrics = manifest.get("metrics", {})

    st.markdown(f"#### {selected.label} · `{selected.version}`")
    st.caption(
        f"Trained on **{selected.dataset_name}** to predict **{selected.target_column}** · "
        f"{manifest.get('duration_seconds', 0):.0f}s · "
        f"{manifest.get('encoded_feature_count', 0)} model inputs"
    )
    render_metric_row(selected, metrics)

    overview, plan_tab, features_tab, models_tab, curves_tab = st.tabs(
        [
            ":material/summarize: Summary",
            ":material/checklist: The plan",
            ":material/functions: Features",
            ":material/leaderboard: Models tried",
            ":material/show_chart: Curves",
        ]
    )

    with overview:
        render_card_overview(selected, manifest, metrics)
    with plan_tab:
        render_card_plan(manifest)
    with features_tab:
        render_card_features(manifest)
    with models_tab:
        render_card_candidates(manifest)
    with curves_tab:
        render_card_curves(metrics)

    st.divider()
    danger_left, danger_right = st.columns([3, 1])
    danger_left.caption(
        "Deleting a version removes its model file permanently. Prediction history is kept."
    )
    if danger_right.button("Delete this version", width="stretch"):
        store.delete(chosen)
        st.rerun()


def render_card_overview(bundle: ModelBundle, manifest: dict, metrics: dict) -> None:
    narrative = manifest.get("narrative", {})
    if narrative.get("summary"):
        st.markdown(narrative["summary"])

    st.markdown("**How this pipeline was designed**")
    try:
        plan = TrainingPlan.model_validate(manifest.get("plan", {}))
        st.info(provenance_summary(plan), icon=":material/smart_toy:")
    except Exception:
        st.caption("Plan provenance is unavailable for this version.")

    summary = manifest.get("dataset_summary", {})
    tiles = st.columns(4)
    tiles[0].metric("Rows supplied", f"{summary.get('rows_supplied', 0):,}")
    tiles[1].metric("Rows used", f"{summary.get('rows_used', 0):,}")
    tiles[2].metric("Trained on", f"{summary.get('train_rows', 0):,}")
    tiles[3].metric("Held out", f"{summary.get('test_rows', 0):,}")

    importance = manifest.get("importance") or []
    if importance:
        st.markdown("**Which columns carry the signal**")
        top = importance[:12]
        figure = px.bar(
            pd.DataFrame(
                {"column": [r["label"] for r in top], "importance": [r["importance"] for r in top]}
            ),
            x="importance", y="column", orientation="h",
            color_discrete_sequence=[NEUTRAL],
        )
        figure.update_layout(**chart_layout(360), showlegend=False)
        figure.update_yaxes(autorange="reversed")
        st.plotly_chart(figure, width="stretch")
        st.caption(
            "How much the headline score drops when a column is shuffled. "
            "A near-zero bar means the model barely uses it."
        )

    matrix = metrics.get("confusion_matrix")
    if matrix:
        st.markdown("**Confusion matrix on held-out data**")
        figure = px.imshow(
            matrix["matrix"],
            x=[f"predicted {label}" for label in matrix["labels"]],
            y=[f"actually {label}" for label in matrix["labels"]],
            text_auto=True,
            color_continuous_scale="Blues",
        )
        figure.update_layout(**chart_layout(320), coloraxis_showscale=False)
        st.plotly_chart(figure, width="stretch")

    library = manifest.get("library_versions", {})
    if library:
        st.caption("Built with " + " · ".join(f"{k} {v}" for k, v in library.items()))


def render_card_plan(manifest: dict) -> None:
    plan = manifest.get("plan", {})
    schema = plan.get("schema_plan", {})
    cleaning = plan.get("cleaning_plan", {})

    if schema.get("summary"):
        st.markdown(schema["summary"])

    st.markdown("**Column roles**")
    rows = [
        {
            "Column": spec.get("name"),
            "Role": str(spec.get("role", "")).replace("_", " "),
            "Shown as": spec.get("display_label"),
            "Why": spec.get("reason") or spec.get("description") or "",
        }
        for spec in schema.get("columns", [])
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", height=320)

    st.markdown("**Cleaning rules**")
    if cleaning.get("notes"):
        st.caption(cleaning["notes"])
    rules = [
        {
            "Column": rule.get("column"),
            "Fill gaps with": rule.get("impute"),
            "Repairs": ", ".join(
                filter(
                    None,
                    [
                        "coerce to number" if rule.get("coerce_numeric") else "",
                        "trim spaces" if rule.get("strip_whitespace") else "",
                        f"clip {rule.get('clip_min')}–{rule.get('clip_max')}"
                        if rule.get("clip_min") is not None or rule.get("clip_max") is not None
                        else "",
                        f"fold levels under {rule.get('rare_category_min_frequency')}"
                        if rule.get("rare_category_min_frequency")
                        else "",
                    ],
                )
            )
            or "—",
            "Why": rule.get("reason", ""),
        }
        for rule in cleaning.get("columns", [])
    ]
    st.dataframe(pd.DataFrame(rules), width="stretch", height=300)

    with st.expander("Stage-by-stage provenance"):
        st.dataframe(pd.DataFrame(plan.get("provenance", [])), width="stretch")


def render_card_features(manifest: dict) -> None:
    from placement_ai.pipeline.dsl import describe_spec
    from placement_ai.plans import FeatureSpec

    features = manifest.get("plan", {}).get("feature_plan", {}).get("features", [])
    if not features:
        st.caption("No derived features were built for this model.")
        return

    notes = manifest.get("plan", {}).get("feature_plan", {}).get("notes")
    if notes:
        st.caption(notes)

    rows = []
    for spec in features:
        try:
            formula = describe_spec(FeatureSpec.model_validate(spec))
        except Exception:
            formula = f"{spec.get('op')}({', '.join(spec.get('inputs', []))})"
        rows.append(
            {
                "Feature": spec.get("name"),
                "How it is computed": formula,
                "Why it was built": spec.get("rationale", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", height=420)
    st.caption(
        f"{len(rows)} feature(s) built on top of the columns you uploaded. Each is a "
        "closed-form transform — the planner chooses from a fixed vocabulary and never writes code."
    )


def render_card_candidates(manifest: dict) -> None:
    candidates = manifest.get("candidates") or []
    if not candidates:
        return
    primary = manifest.get("primary_metric", "roc_auc").replace("_", " ")
    champion = (manifest.get("champion") or {}).get("algorithm")

    rows = [
        {
            "Model": c["label"] + ("  ← chosen" if c["algorithm"] == champion else ""),
            f"Held-out {primary}": round(c.get("score", 0), 4),
            "Cross-validated": (
                "—" if c.get("cv_mean") is None or pd.isna(c.get("cv_mean"))
                else f"{c['cv_mean']:.4f} ± {c.get('cv_std', 0):.4f}"
            ),
            "Train time": f"{c.get('train_seconds', 0):.1f}s",
            "Ensemble weight": c.get("ensemble_weight"),
            "Why it was tried": c.get("rationale", ""),
        }
        for c in candidates
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", height=280)

    notes = manifest.get("plan", {}).get("model_plan", {}).get("notes")
    if notes:
        st.caption(notes)
    st.caption(
        "The ensemble averages the other models using the planner's weights. It is kept "
        "only when it genuinely outscores every single model."
    )


def render_card_curves(metrics: dict) -> None:
    roc = metrics.get("roc_curve")
    pr = metrics.get("pr_curve")
    if not roc and not pr:
        st.caption("Curves are only produced for a two-outcome target.")
        return

    left, right = st.columns(2)
    if roc:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                    line={"color": NEUTRAL, "width": 3}, name="Model"))
        figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                    line={"color": MUTED, "dash": "dash"}, name="Random guessing"))
        figure.update_layout(**chart_layout(340, "ROC curve"))
        left.plotly_chart(figure, width="stretch")
    if pr:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines",
                                    line={"color": POSITIVE, "width": 3}, name="Model"))
        base = metrics.get("base_rate")
        if base:
            figure.add_hline(y=base, line_dash="dash", line_color=MUTED,
                             annotation_text="base rate")
        figure.update_layout(**chart_layout(340, "Precision–recall curve"))
        right.plotly_chart(figure, width="stretch")

    tiles = st.columns(3)
    for column, key, label in (
        (tiles[0], "roc_auc", "ROC-AUC"),
        (tiles[1], "average_precision", "Average precision"),
        (tiles[2], "brier_score", "Brier score (lower is better)"),
    ):
        value = metrics.get(key)
        column.metric(label, f"{value:.4f}" if isinstance(value, (int, float)) else "—")


# =============================================================================
# 9. CHART BUILDERS
# =============================================================================
def probability_gauge(probability: float, threshold: float, positive: str | None) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": f"Probability of {positive}" if positive else "Confidence"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED},
                "bar": {"color": POSITIVE if probability >= threshold else NEGATIVE},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, threshold * 100], "color": "rgba(248,113,113,0.15)"},
                    {"range": [threshold * 100, 100], "color": "rgba(52,211,153,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#F8FAFC", "width": 3},
                    "value": threshold * 100,
                },
            },
        )
    )
    figure.update_layout(**chart_layout(260))
    return figure


def driver_chart(drivers: list[dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(drivers).sort_values("delta")
    figure = go.Figure(
        go.Bar(
            x=frame["delta"],
            y=frame["label"],
            orientation="h",
            marker_color=[POSITIVE if d > 0 else NEGATIVE for d in frame["delta"]],
            customdata=frame[["value", "typical"]],
            hovertemplate="<b>%{y}</b><br>this record: %{customdata[0]}"
            "<br>typical: %{customdata[1]}<br>effect: %{x:+.3f}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_color=MUTED)
    figure.update_layout(**chart_layout(max(260, 34 * len(frame))), showlegend=False)
    return figure


# =============================================================================
# 10. MAIN
# =============================================================================
def main() -> None:
    workspace = render_sidebar()

    st.title("Adaptive placement intelligence")
    st.caption(
        "Upload your own records, let the AI planner design and train a model on them, "
        "then use that model for as long as you like — retraining only when you choose to."
    )

    if workspace is None:
        render_welcome()
        return

    bundle = ModelStore(workspace).active()

    train_tab, predict_tab, history_tab, model_tab = st.tabs(
        [
            ":material/model_training: Train",
            ":material/query_stats: Predict",
            ":material/history: History",
            ":material/description: Model card",
        ]
    )
    with train_tab:
        render_train_tab(workspace)
    with predict_tab:
        render_predict_tab(workspace, bundle)
    with history_tab:
        render_history_tab(workspace, bundle)
    with model_tab:
        render_model_tab(workspace, bundle)


def render_welcome() -> None:
    st.info(
        "Create a workspace in the sidebar to begin. Each workspace keeps its own data, "
        "its own models and its own prediction history.",
        icon=":material/arrow_back:",
    )

    left, middle, right = st.columns(3)
    with left:
        st.markdown("##### :material/upload_file: 1. Upload")
        st.markdown(
            "Bring any spreadsheet with one column recording the outcome you care about. "
            "Column names, units and layout are yours to choose."
        )
    with middle:
        st.markdown("##### :material/smart_toy: 2. Let it plan and train")
        st.markdown(
            "The AI planner reads your columns, decides how to clean them, designs new "
            "features, picks the models and weights them. Training may take a few minutes."
        )
    with right:
        st.markdown("##### :material/query_stats: 3. Predict")
        st.markdown(
            "Your model is saved to your workspace. Use it one record at a time or on a "
            "whole file, for as long as you like, until you decide to retrain."
        )

    if not Path(SAMPLE_DATASET_PATH).exists():
        return
    st.caption(
        "No data to hand? A 10,000-row sample of student placement records ships with "
        "the app — you can train on it from the Train tab."
    )


main()
