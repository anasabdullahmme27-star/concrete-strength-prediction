"""Concrete Strength Predictor — single-screen edition.

One view, no scrolling: enter a mix, press the button, read the number. A
second tab keeps a persistent tally of how the models are being used.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Concrete Compressive Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# numpy and pandas are imported for their side effect: Plotly's validators take
# whatever pandas is already in sys.modules and touch pd.Series directly, so a
# half-imported module on another Streamlit thread would crash the app. This
# blocks on the import lock until the module is complete.
import numpy  # noqa: E402, F401
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from src import config, predictor, stats  # noqa: E402

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
BG = "#0e161f"
PANEL = "#151f2b"
PANEL_2 = "#1b2733"
BORDER = "#243444"
INK = "#e8eef5"
INK_SOFT = "#9fb0c2"
INK_MUTED = "#6b7e92"

TEAL = "#00d9a3"
BLUE = "#3b9ef5"
VIOLET = "#b06bf5"
AMBER = "#f5a623"
RED = "#ff6b6b"

CARD_ACCENTS = (TEAL, BLUE, VIOLET, TEAL, BLUE, VIOLET)

st.markdown(
    f"""
    <style>
    .stApp {{ background: {BG}; }}
    /* The header is a transparent full-width bar that sits on top of the
       first row of content, so it swallows clicks meant for the tabs. Let
       pointer events fall through it, and give them back to the toolbar. */
    header[data-testid="stHeader"] {{
        height: 2.1rem; background: transparent; pointer-events: none;
    }}
    /* The toolbar and the bar inside it both stretch the full width even
       though the buttons sit in a 200px cluster on the right, so every level
       above that cluster has to pass clicks through. */
    header[data-testid="stHeader"] [data-testid="stToolbar"],
    header[data-testid="stHeader"] [data-testid="stToolbar"] > * {{
        pointer-events: none;
    }}
    header[data-testid="stHeader"] [data-testid="stToolbar"] > * > * {{
        pointer-events: auto;
    }}
    .block-container {{
        padding: 0.3rem 1.4rem 0.4rem; max-width: 100%;
    }}
    #MainMenu, footer {{ visibility: hidden; }}

    /* One screen, no scrolling: tighten the default stacking gap. */
    div[data-testid="stVerticalBlock"] {{ gap: 0.3rem; }}
    div[data-testid="stHorizontalBlock"] {{ gap: 0.55rem; }}
    div[data-testid="stElementContainer"] {{ margin: 0; }}
    div[data-testid="stNumberInput"] {{ margin: 0; }}
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{ gap: 0.3rem; }}

    h1, h2, h3, h4, p, label, span, div {{ color: {INK}; }}

    /* header ------------------------------------------------------------ */
    .app-title {{
        font-size: 1.5rem !important; font-weight: 700; letter-spacing: -0.02em;
        color: {INK}; line-height: 1.15; margin: 0 0 0.05rem;
    }}
    .app-sub {{ font-size: 0.76rem !important; color: {INK_MUTED}; margin: 0; }}
    .badges {{ display: flex; gap: 0.5rem; justify-content: flex-end; padding-top: 0.5rem; }}
    .badge {{
        border-radius: 7px; padding: 0.34rem 0.7rem; font-size: 0.78rem;
        font-weight: 700; color: #06231c; white-space: nowrap;
    }}
    .badge .k {{ opacity: 0.72; font-weight: 700; margin-right: 0.28rem; }}

    /* panels ------------------------------------------------------------ */
    .panel {{
        background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
        padding: 0.55rem 0.75rem 0.65rem; margin-bottom: 0.5rem;
    }}
    .panel-title {{
        font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.13em;
        font-weight: 700; color: {INK_MUTED}; margin: 0 0 0.4rem;
    }}
    .ing-label {{
        font-size: 0.86rem !important; color: {INK}; font-weight: 500; margin: 0;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .ing-label .ico {{ color: {TEAL}; margin-right: 0.4rem; }}
    .ing-label .unit {{
        color: {INK_MUTED}; font-size: 0.72rem; margin-left: 0.3rem;
    }}
    .ing-label.is-out .unit {{ color: {AMBER}; }}

    /* result ------------------------------------------------------------ */
    .result-panel {{
        background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
        padding: 0.5rem 1rem 0.4rem; margin-bottom: 0.5rem;
    }}
    .result-label {{
        font-size: 0.85rem !important; color: {INK_SOFT}; margin: 0 0 0.1rem;
    }}
    .result-value {{
        font-size: 3.2rem !important; font-weight: 700; color: {VIOLET};
        line-height: 1.05; margin: 0 0 0.55rem; letter-spacing: -0.02em;
    }}
    .result-grade-label {{ font-size: 0.85rem !important; color: {INK_SOFT}; margin: 0 0 0.1rem; }}
    .result-grade {{
        font-size: 1.2rem !important; font-weight: 700; color: {VIOLET}; margin: 0 0 0.15rem;
    }}
    .result-note {{ font-size: 0.8rem !important; color: {INK_MUTED}; margin: 0; }}

    /* stat cards --------------------------------------------------------- */
    .stat {{
        background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
        border-top: 3px solid var(--accent); padding: 0.6rem 0.85rem 0.65rem;
        height: 100%; min-height: 118px;
    }}
    .stat .head {{ font-size: 0.76rem !important; color: {INK_SOFT}; margin: 0 0 0.2rem; }}
    .stat .head .ico {{ margin-right: 0.35rem; }}
    .stat .val {{
        font-size: 1.35rem !important; font-weight: 700; color: var(--accent);
        line-height: 1.1; margin: 0 0 0.12rem;
    }}
    .stat .sub {{ font-size: 0.69rem !important; color: {INK_MUTED}; margin: 0; }}

    /* status bar --------------------------------------------------------- */
    .status {{
        background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
        padding: 0.35rem 0.8rem; font-size: 0.78rem !important; color: {TEAL};
        margin-top: 0.15rem;
    }}
    .status.is-idle {{ color: {INK_MUTED}; }}
    .status .warn {{ color: {AMBER}; }}

    /* widgets ------------------------------------------------------------ */
    div[data-testid="stNumberInput"] input {{
        background: {PANEL_2}; color: {INK}; border-radius: 6px;
        text-align: right; font-weight: 600; font-size: 0.92rem;
        padding: 0.22rem 0.5rem; font-variant-numeric: tabular-nums;
    }}
    div[data-testid="stNumberInput"] button {{ background: {PANEL_2}; color: {INK_SOFT}; }}
    div[data-testid="stNumberInput"] button:hover {{ color: {TEAL}; }}
    div[data-baseweb="select"] > div {{
        background: {PANEL_2}; border-color: {BORDER}; font-size: 0.84rem;
    }}
    div[data-testid="stSelectbox"] label p {{ font-size: 0.7rem !important; color: {INK_MUTED}; }}

    .stButton > button {{
        border-radius: 7px; font-weight: 600; font-size: 0.82rem;
        background: {PANEL_2}; color: {INK_SOFT}; border: 1px solid {BORDER};
        padding: 0.3rem 0.5rem;
    }}
    .stButton > button:hover {{ border-color: {TEAL}; color: {TEAL}; }}
    .stButton > button[kind="primary"] {{
        background: {TEAL}; color: #06231c; border: none;
        font-size: 1rem; font-weight: 700; letter-spacing: 0.04em;
        padding: 0.55rem 0.5rem;
    }}
    .stButton > button[kind="primary"]:hover {{ background: #17e9b4; color: #06231c; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 0.4rem; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{ font-size: 0.85rem; font-weight: 600; padding: 0.3rem 0.7rem; }}
    .stTabs [aria-selected="true"] {{ color: {TEAL} !important; }}

    div[data-testid="stMetric"] {{
        background: {PANEL}; border: 1px solid {BORDER};
        border-radius: 9px; padding: 0.6rem 0.8rem;
    }}
    div[data-testid="stMetricLabel"] p {{ font-size: 0.72rem !important; color: {INK_MUTED}; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.35rem; color: {INK}; }}
    div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}
    hr {{ margin: 0.4rem 0; border-color: {BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gauge(value: float) -> go.Figure:
    """The dial. Bands are EN 206 classes; the needle is this mix.

    Colour is deliberately redundant here — the value is printed beside the
    dial and the class is spelled out, so the bands are an at-a-glance guide
    rather than the only way to read the result.
    """
    bands = [
        (0, 20, VIOLET),
        (20, 40, BLUE),
        (40, 60, TEAL),
        (60, 80, AMBER),
        (80, 100, RED),
    ]
    figure = go.Figure(
        go.Indicator(
            mode="gauge",
            value=max(0.0, min(value, 100.0)),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickwidth=1,
                    tickcolor=INK_MUTED,
                    tickvals=[0, 20, 40, 60, 80, 100],
                    tickfont=dict(size=11, color=INK_MUTED),
                ),
                bar=dict(color="rgba(0,0,0,0)", thickness=0),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[dict(range=[low, high], color=colour) for low, high, colour in bands],
                # Plotly's angular gauge has no centre needle, so the marker is
                # a full-depth radial line across the band instead.
                threshold=dict(
                    line=dict(color="#ffffff", width=6), thickness=1.0, value=value
                ),
            ),
        )
    )
    figure.update_layout(
        height=248,
        margin=dict(l=34, r=34, t=10, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif"),
    )
    return figure


def stat_card(icon: str, title: str, value: str, note: str, accent: str) -> str:
    return (
        f'<div class="stat" style="--accent:{accent}">'
        f'<p class="head"><span class="ico">{icon}</span>{title}</p>'
        f'<p class="val">{value}</p>'
        f'<p class="sub">{note}</p>'
        "</div>"
    )


def _render_ingredients(active: list[config.Feature]) -> None:
    """One row per ingredient: name and unit on the left, typed value on the right.

    The label is drawn as markdown rather than as the widget's own label so the
    row stays on a single line, which is what keeps nine inputs on one screen.
    """
    for feature in active:
        low, high = config.feature_range(feature)
        label_column, input_column = st.columns([1.0, 1.0], vertical_alignment="center")
        value = float(
            input_column.number_input(
                feature.label,
                min_value=0.0,
                max_value=round(high * 2.5, feature.decimals),
                step=feature.step,
                format=f"%.{feature.decimals}f",
                key=f"in_{feature.key}",
                label_visibility="collapsed",
            )
        )
        outside = value < low or value > high
        label_column.markdown(
            f'<p class="ing-label{" is-out" if outside else ""}">'
            f'<span class="ico">{feature.icon}</span>{feature.label}'
            f'<span class="unit">{feature.unit}</span></p>',
            unsafe_allow_html=True,
        )


def current_values() -> dict[str, float]:
    return {
        feature.column: float(st.session_state.get(f"in_{feature.key}", config.feature_median(feature)))
        for feature in config.FEATURES
    }


def apply_preset(name: str) -> None:
    for column, value in config.PRESETS[name].items():
        feature = next(f for f in config.FEATURES if f.column == column)
        st.session_state[f"in_{feature.key}"] = float(value)


def clear_inputs() -> None:
    for feature in config.FEATURES:
        st.session_state[f"in_{feature.key}"] = 0.0 if feature.key != "age" else 1.0
    st.session_state.pop("result", None)


if "initialised" not in st.session_state:
    apply_preset("High Performance")
    st.session_state["initialised"] = True


tab_predict, tab_analytics = st.tabs(["⚡  Predictor", "📊  Analytics"])

# ===========================================================================
# Predictor
# ===========================================================================
with tab_predict:
    head = st.columns([3.9, 1.75, 1.75, 3.6], vertical_alignment="center")
    title_slot = head[0].empty()

    with head[1]:
        family = st.selectbox(
            "Feature set",
            options=list(config.FAMILIES),
            format_func=lambda key: config.FAMILIES[key]["label"],
            key="family",
        )
    candidates = predictor.available_models(family)
    with head[2]:
        spec = st.selectbox(
            "Model",
            options=candidates,
            format_func=lambda s: f"{s.name}{' ★' if s.is_champion else ''}",
            key=f"model_{family}",
        )

    title_slot.markdown(
        f'<p class="app-title">🏗️ Concrete Strength Predictor</p>'
        f'<p class="app-sub">Machine Learning · {spec.name} · '
        f'{config.FAMILIES[family]["inputs"]} inputs · UCI Dataset</p>',
        unsafe_allow_html=True,
    )
    with head[3]:
        st.markdown(
            '<div class="badges">'
            f'<span class="badge" style="background:{TEAL}"><span class="k">R²</span>{spec.r2:.4f}</span>'
            f'<span class="badge" style="background:{BLUE}"><span class="k">RMSE</span>{spec.rmse:.4f} MPa</span>'
            f'<span class="badge" style="background:{AMBER}"><span class="k">MAE</span>{spec.mae:.4f} MPa</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    active = [f for f in config.FEATURES if f.column in spec.columns]

    left, right = st.columns([1.02, 2.0], gap="medium")

    # --- inputs -----------------------------------------------------------
    with left:
        with st.container(border=True):
            st.markdown('<p class="panel-title">Quick Presets</p>', unsafe_allow_html=True)
            preset_names = list(config.PRESETS)
            for row_start in (0, 3):
                preset_columns = st.columns(3, gap="small")
                for column, name in zip(preset_columns, preset_names[row_start:row_start + 3]):
                    if column.button(name, key=f"preset_{name}", width="stretch"):
                        apply_preset(name)
                        st.rerun()

        with st.container(border=True):
            st.markdown('<p class="panel-title">Mix Ingredients</p>', unsafe_allow_html=True)
            _render_ingredients(active)

        run = st.button("⚡  PREDICT STRENGTH", type="primary", width="stretch")
        if st.button("Clear All", key="clear_all", width="stretch"):
            clear_inputs()
            st.rerun()

    values = current_values()

    if run:
        prediction = predictor.predict(spec, values)
        stats.record(spec.key, spec.name, family, prediction)
        st.session_state["result"] = {
            "value": prediction,
            "model": spec.name,
            "committed": True,
        }

    result = st.session_state.get("result")
    if result is None or result.get("model") != spec.name:
        # Show a live figure straight away rather than an empty panel; only an
        # explicit button press is counted in the statistics.
        result = {
            "value": predictor.predict(spec, values),
            "model": spec.name,
            "committed": False,
        }

    prediction = result["value"]
    grade, grade_note = config.strength_class(prediction)
    ratios = predictor.indicators(values)
    flagged = predictor.out_of_range(values, spec.columns)

    # --- result -----------------------------------------------------------
    with right:
        with st.container(border=True):
            gauge_column, value_column = st.columns([0.82, 1.18], vertical_alignment="center")
            with gauge_column:
                st.plotly_chart(
                    gauge(prediction),
                    width="stretch",
                    config={"displayModeBar": False, "staticPlot": True},
                )
            with value_column:
                st.markdown(
                    f'<p class="result-label">Predicted Strength</p>'
                    f'<p class="result-value">{prediction:.2f} MPa</p>'
                    f'<p class="result-grade-label">Concrete Grade</p>'
                    f'<p class="result-grade">{grade}</p>'
                    f'<p class="result-note">{grade_note}</p>',
                    unsafe_allow_html=True,
                )

        cards = [
            ("💧", "Water / Cement", f"{ratios['wc_ratio']:.4f}",
             "Excellent" if ratios["wc_ratio"] < 0.40
             else "Good" if ratios["wc_ratio"] < 0.50
             else "Moderate" if ratios["wc_ratio"] < 0.60 else "High"),
            ("▦", "Total Binder", f"{ratios['total_binder']:.1f} kg/m³",
             f"Cement {values[config.CEMENT]:.0f} + Slag {values[config.SLAG]:.0f}"
             f" + Fly Ash {values[config.FLY_ASH]:.0f}"),
            ("📊", "Binder Ratio", f"{ratios['binder_ratio']:.4f}",
             "Binder ÷ (Binder + Water)"),
            ("◍", "Total Aggregate", f"{ratios['total_aggregate']:.1f} kg/m³",
             f"Coarse {values[config.COARSE_AGG]:.0f} + Fine {values[config.FINE_AGG]:.0f}"),
            ("🧪", "Paste Volume", f"{ratios['paste_volume']:.4f}",
             "(C + S + FA + W) ÷ 2400"),
            ("🏅", "Strength Class", grade.split(" —")[0], grade_note),
        ]
        for row_start in (0, 3):
            card_columns = st.columns(3, gap="small")
            for offset, (column, card) in enumerate(
                zip(card_columns, cards[row_start:row_start + 3])
            ):
                icon, title, value_text, note = card
                column.markdown(
                    stat_card(icon, title, value_text, note, CARD_ACCENTS[row_start + offset]),
                    unsafe_allow_html=True,
                )

    # --- status bar -------------------------------------------------------
    if flagged:
        status = (
            f'<span class="warn">⚠ Outside training range: {", ".join(flagged)}</span>'
            f' · W/C = {ratios["wc_ratio"]:.3f} · Binder = {ratios["total_binder"]:.0f} kg/m³'
        )
        status_class = "status"
    elif result["committed"]:
        status = (
            f'✓ Prediction complete · W/C = {ratios["wc_ratio"]:.3f}'
            f' · Binder = {ratios["total_binder"]:.0f} kg/m³'
        )
        status_class = "status"
    else:
        status = (
            f'Live preview · press PREDICT STRENGTH to record this run'
            f' · W/C = {ratios["wc_ratio"]:.3f}'
        )
        status_class = "status is-idle"
    st.markdown(f'<div class="{status_class}">{status}</div>', unsafe_allow_html=True)


# ===========================================================================
# Analytics
# ===========================================================================
with tab_analytics:
    payload = stats.load()
    totals = stats.overall(payload)
    rows = stats.per_model(payload)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total predictions", f"{totals['total']:,}")
    k2.metric("Models used", f"{totals['models_used']}")
    k3.metric(
        "Mean predicted strength",
        f"{totals['mean']:.2f} MPa" if rows else "—",
    )
    k4.metric(
        "Last prediction",
        f"{totals['last_value']:.2f} MPa" if totals["last_value"] is not None else "—",
    )
    k5.metric("Recording since", totals["since"] or "—")

    if not rows:
        st.info(
            "No predictions recorded yet. Run one from the Predictor tab and this "
            "tab fills up. Counters are stored on disk and survive a restart."
        )
    else:
        frame = pd.DataFrame(rows)

        chart_column, table_column = st.columns([1.15, 1.0], gap="medium")

        with chart_column:
            figure = go.Figure(
                go.Bar(
                    x=frame["Runs"],
                    y=frame["Model"] + "  ·  " + frame["Family"],
                    orientation="h",
                    width=0.5,
                    marker=dict(color=TEAL, line=dict(width=0)),
                    text=frame["Runs"],
                    textposition="outside",
                    textfont=dict(color=INK_SOFT, size=12),
                    hovertemplate="%{y}<br>%{x} runs<extra></extra>",
                )
            )
            figure.update_layout(
                title=dict(
                    text="Runs per model", font=dict(size=14, color=INK), x=0, xanchor="left"
                ),
                height=max(240, 60 + 46 * len(frame)),
                margin=dict(l=8, r=26, t=44, b=8),
                paper_bgcolor=PANEL,
                plot_bgcolor=PANEL,
                font=dict(family="Inter, 'Segoe UI', sans-serif", color=INK_SOFT, size=12),
                xaxis=dict(
                    gridcolor=BORDER, zeroline=False, title="predictions",
                    range=[0, float(frame["Runs"].max()) * 1.35],
                    dtick=max(1, int(frame["Runs"].max() // 5)),
                ),
                yaxis=dict(gridcolor=BORDER, zeroline=False),
                bargap=0.4,
                showlegend=False,
            )
            st.plotly_chart(
                figure, width="stretch", config={"displayModeBar": False}
            )

        with table_column:
            figure = go.Figure(
                go.Bar(
                    x=frame["Mean MPa"],
                    y=frame["Model"] + "  ·  " + frame["Family"],
                    orientation="h",
                    width=0.5,
                    marker=dict(color=BLUE, line=dict(width=0)),
                    text=[f"{v:.1f}" for v in frame["Mean MPa"]],
                    textposition="outside",
                    textfont=dict(color=INK_SOFT, size=12),
                    hovertemplate="%{y}<br>mean %{x:.2f} MPa<extra></extra>",
                )
            )
            figure.update_layout(
                title=dict(
                    text="Mean predicted strength per model",
                    font=dict(size=14, color=INK), x=0, xanchor="left",
                ),
                height=max(240, 60 + 46 * len(frame)),
                margin=dict(l=8, r=26, t=44, b=8),
                paper_bgcolor=PANEL,
                plot_bgcolor=PANEL,
                font=dict(family="Inter, 'Segoe UI', sans-serif", color=INK_SOFT, size=12),
                xaxis=dict(
                    gridcolor=BORDER, zeroline=False, title="MPa",
                    range=[0, float(frame["Mean MPa"].max()) * 1.35],
                ),
                yaxis=dict(gridcolor=BORDER, zeroline=False),
                bargap=0.4,
                showlegend=False,
            )
            st.plotly_chart(
                figure, width="stretch", config={"displayModeBar": False}
            )

        st.dataframe(
            frame.drop(columns=["key"]),
            width="stretch",
            hide_index=True,
            height=42 + 35 * len(frame),
        )

        if st.button("Reset statistics", key="reset_stats"):
            stats.reset()
            st.rerun()
