import sys
import traceback

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import dash
from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from data import load_data

# ── Palette ──────────────────────────────────────────────────────────────────

DECADE_COLORS = {
    "1960s": "#3b82f6",
    "1970s": "#f59e0b",
    "1980s": "#ef4444",
    "1990s": "#10b981",
    "2000s": "#8b5cf6",
    "2010s": "#06b6d4",
    "2020s": "#f97316",
}

UNEMPLOYMENT_COLOR = "#3b82f6"
INFLATION_COLOR = "#ef4444"

BACKGROUND = "#1a2a3f"
SURFACE = "#243548"
BORDER = "#3d536e"
TEXT_PRIMARY = "#f1f5f9"
TEXT_MUTED = "#94a3b8"

FONT_FAMILY = "'Inter', 'Helvetica Neue', Arial, sans-serif"

AXIS_STYLE = dict(
    gridcolor=BORDER,
    linecolor=BORDER,
    tickfont=dict(color=TEXT_MUTED),
    title_font=dict(color=TEXT_MUTED),
)

LEGEND_STYLE = dict(
    bgcolor="rgba(30,41,59,0.8)",
    bordercolor=BORDER,
    borderwidth=1,
    font=dict(color=TEXT_PRIMARY),
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BACKGROUND,
    plot_bgcolor=SURFACE,
    font=dict(family=FONT_FAMILY, color=TEXT_PRIMARY),
    margin=dict(l=60, r=40, t=60, b=60),
    hoverlabel=dict(
        bgcolor=SURFACE,
        bordercolor=BORDER,
        font=dict(family=FONT_FAMILY, color=TEXT_PRIMARY),
    ),
)


# ── Chart builders ────────────────────────────────────────────────────────────

def build_time_series(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=[""],
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["unemployment"],
            name="Unemployment Rate",
            line=dict(color=UNEMPLOYMENT_COLOR, width=1.8),
            hovertemplate="<b>%{x|%b %Y}</b><br>Unemployment: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["inflation"],
            name="CPI Inflation (YoY)",
            line=dict(color=INFLATION_COLOR, width=1.8),
            hovertemplate="<b>%{x|%b %Y}</b><br>Inflation: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    # Highlight recessions with subtle bands (NBER approximate dates)
    recessions = [
        ("1960-04-01", "1961-02-01"),
        ("1969-12-01", "1970-11-01"),
        ("1973-11-01", "1975-03-01"),
        ("1980-01-01", "1980-07-01"),
        ("1981-07-01", "1982-11-01"),
        ("1990-07-01", "1991-03-01"),
        ("2001-03-01", "2001-11-01"),
        ("2007-12-01", "2009-06-01"),
        ("2020-02-01", "2020-04-01"),
    ]
    for start, end in recessions:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(148,163,184,0.07)",
            layer="below",
            line_width=0,
        )

    fig.add_hline(
        y=0, secondary_y=True,
        line=dict(color=BORDER, width=1, dash="dot"),
    )

    layout = dict(
        **PLOTLY_LAYOUT,
        title=dict(
            text="U.S. Unemployment & Inflation (1960–Present)",
            font=dict(size=18, color=TEXT_PRIMARY),
            x=0.03,
        ),
        hovermode="x unified",
        legend=dict(**LEGEND_STYLE, orientation="h", y=-0.12, x=0),
    )
    fig.update_layout(**layout)

    fig.update_yaxes(title_text="Unemployment Rate (%)", secondary_y=False, **AXIS_STYLE)
    fig.update_yaxes(title_text="CPI Inflation YoY (%)", secondary_y=True, zeroline=False, **AXIS_STYLE)
    fig.update_xaxes(**AXIS_STYLE)

    return fig


def _hyperbolic_curve(x_vals, y_vals, n=200):
    """Fit π = a/u + b and return (x_range, y_range) for plotting."""
    a, b = np.polyfit(1.0 / x_vals, y_vals, 1)
    x_range = np.linspace(x_vals.min(), x_vals.max(), n)
    return x_range, a / x_range + b


def _event_traces(df, window_start, window_end, pick, text, dx, dy, legendgroup):
    """
    Return (arrow_line_trace, label_trace) for an event annotation.
    Both share legendgroup so they hide/show with the decade toggle.
    dx/dy are offsets in data coordinates from the anchor point to the label.
    """
    window = df.loc[window_start:window_end]
    if window.empty:
        return []
    row = window.loc[window["inflation"].idxmax() if pick == "max_inflation"
                     else window["unemployment"].idxmax()]
    x0, y0 = row["unemployment"], row["inflation"]
    x1, y1 = x0 + dx, y0 + dy

    arrow = go.Scatter(
        x=[x0, x1], y=[y0, y1],
        mode="lines",
        line=dict(color=TEXT_MUTED, width=1),
        legendgroup=legendgroup,
        showlegend=False,
        hoverinfo="skip",
    )
    label = go.Scatter(
        x=[x1], y=[y1],
        mode="text",
        text=[text],
        textfont=dict(family=FONT_FAMILY, size=10, color=TEXT_PRIMARY),
        textposition="middle center",
        legendgroup=legendgroup,
        showlegend=False,
        hoverinfo="skip",
    )
    return [arrow, label]


def build_phillips_curve(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # ── 1. Decade scatter dots ────────────────────────────────────────────────
    for decade, color in DECADE_COLORS.items():
        subset = df[df["decade"] == decade]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["unemployment"],
                y=subset["inflation"],
                mode="markers",
                name=decade,
                legendgroup=decade,
                legendgrouptitle=dict(text="Decade", font=dict(color=TEXT_MUTED, size=11))
                if decade == "1960s" else None,
                marker=dict(color=color, size=5, opacity=0.7, line=dict(width=0)),
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    "Unemployment: %{x:.1f}%<br>"
                    "Inflation: %{y:.1f}%<extra></extra>"
                ),
                customdata=subset.index.strftime("%b %Y"),
            )
        )

    # ── 2. LOWESS smooth curve through all data ───────────────────────────────
    sorted_df = df.sort_values("unemployment")
    smoothed = lowess(sorted_df["inflation"], sorted_df["unemployment"], frac=0.25, it=3)
    fig.add_trace(
        go.Scatter(
            x=smoothed[:, 0],
            y=smoothed[:, 1],
            mode="lines",
            name="Overall trend (LOWESS)",
            legendgroup="lowess",
            legendgrouptitle=dict(text="Trend Lines", font=dict(color=TEXT_MUTED, size=11)),
            line=dict(color="#e2e8f0", width=2.5, dash="dash"),
            hoverinfo="skip",
        )
    )

    # ── 3. Three era trend lines — tells WHY the curve changed ───────────────
    eras = [
        (
            df[df["decade"] == "1960s"],
            "Classic Phillips Curve",
            "#a78bfa",
        ),
        (
            df[df["decade"].isin(["1970s", "1980s"])],
            "Stagflation Era — relationship breaks",
            "#fb923c",
        ),
        (
            df[df["decade"].isin(["1990s", "2000s", "2010s", "2020s"])],
            "Modern Era — relationship flat",
            "#34d399",
        ),
    ]
    for era_df, name, color in eras:
        x_r, y_r = _hyperbolic_curve(era_df["unemployment"].values, era_df["inflation"].values)
        fig.add_trace(
            go.Scatter(
                x=x_r, y=y_r,
                mode="lines",
                name=name,
                legendgroup=name,
                line=dict(color=color, width=2.5),
                hoverinfo="skip",
            )
        )

    # ── 4. Event annotations as scatter traces (toggle with their decade) ─────
    # dx/dy are data-coordinate offsets for the label relative to the anchor.
    event_specs = [
        ("1973-10", "1975-03", "max_inflation",
         "<b>1970s Oil Crisis</b><br>Stagflation: high unemployment<br>AND high inflation",
         2.2, 2.8, "1970s"),
        ("2008-09", "2010-06", "max_unemployment",
         "<b>2008 Financial Crisis</b><br>Unemployment → 10%<br>Inflation collapses",
         -3.0, -3.2, "2000s"),
        ("2021-06", "2022-12", "max_inflation",
         "<b>Post-COVID 2021–22</b><br>Supply shock: inflation → 9%<br>while unemployment was falling",
         2.0, 1.8, "2020s"),
    ]
    for spec in event_specs:
        for trace in _event_traces(df, *spec):
            fig.add_trace(trace)

    fig.add_hline(y=0, line=dict(color=BORDER, width=1, dash="dot"))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=(
                "The Flattening Phillips Curve: "
                "A 60-Year Story of Changing Economic Relationships"
                "<br><span style='font-size:12px; color:#94a3b8; font-weight:400'>"
                "Each dot represents one month of US economic data. Colored by decade."
                "</span>"
            ),
            font=dict(size=18, color=TEXT_PRIMARY),
            x=0.03,
        ),
        xaxis=dict(**AXIS_STYLE, title="Unemployment Rate (%)"),
        yaxis=dict(**AXIS_STYLE, title="CPI Inflation YoY (%)"),
        legend=dict(**LEGEND_STYLE, groupclick="togglegroup"),
        hovermode="closest",
    )

    return fig


# ── Layout helpers ────────────────────────────────────────────────────────────

def stat_card(label: str, value: str, delta: str | None = None) -> html.Div:
    delta_el = (
        html.Span(delta, style={"fontSize": "12px", "color": TEXT_MUTED, "marginLeft": "8px"})
        if delta else html.Span()
    )
    return html.Div(
        [
            html.P(label, style={"margin": "0 0 4px", "fontSize": "11px",
                                  "textTransform": "uppercase", "letterSpacing": "0.08em",
                                  "color": TEXT_MUTED}),
            html.Div(
                [html.Span(value, style={"fontSize": "24px", "fontWeight": "600",
                                          "color": TEXT_PRIMARY}),
                 delta_el],
                style={"display": "flex", "alignItems": "baseline"},
            ),
        ],
        style={
            "background": SURFACE,
            "border": f"1px solid {BORDER}",
            "borderRadius": "8px",
            "padding": "16px 20px",
            "flex": "1",
        },
    )


def build_kpi_row(df: pd.DataFrame) -> html.Div:
    latest = df.iloc[-1]
    latest_date = df.index[-1].strftime("%b %Y")
    peak_inflation = df["inflation"].max()
    peak_inflation_date = df["inflation"].idxmax().strftime("%Y")
    peak_unemployment = df["unemployment"].max()
    peak_unemployment_date = df["unemployment"].idxmax().strftime("%Y")

    return html.Div(
        [
            stat_card("Unemployment Rate", f"{latest['unemployment']:.1f}%", latest_date),
            stat_card("CPI Inflation (YoY)", f"{latest['inflation']:.1f}%", latest_date),
            stat_card("Peak Inflation", f"{peak_inflation:.1f}%", peak_inflation_date),
            stat_card("Peak Unemployment", f"{peak_unemployment:.1f}%", peak_unemployment_date),
        ],
        style={"display": "flex", "gap": "16px", "marginBottom": "24px"},
    )


# ── App ───────────────────────────────────────────────────────────────────────

print("Fetching data from FRED...", flush=True)
try:
    df = load_data()
    print(f"Loaded {len(df)} observations ({df.index[0].year}–{df.index[-1].year})", flush=True)
except Exception as exc:
    print(f"\nERROR: {exc}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

app = dash.Dash(
    __name__,
    title="Macro Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # expose for production WSGI servers

tab_style = {
    "backgroundColor": SURFACE,
    "color": TEXT_MUTED,
    "border": f"1px solid {BORDER}",
    "borderRadius": "6px 6px 0 0",
    "padding": "10px 20px",
    "fontFamily": FONT_FAMILY,
    "fontSize": "13px",
    "fontWeight": "500",
}
tab_selected_style = {
    **tab_style,
    "backgroundColor": BACKGROUND,
    "color": TEXT_PRIMARY,
    "borderBottom": f"2px solid {UNEMPLOYMENT_COLOR}",
}

app.layout = html.Div(
    style={
        "backgroundColor": BACKGROUND,
        "minHeight": "100vh",
        "fontFamily": FONT_FAMILY,
        "color": TEXT_PRIMARY,
        "padding": "32px 40px",
    },
    children=[
        # Header
        html.Div(
            [
                html.Div(
                    [
                        html.H1(
                            "U.S. Macroeconomic Dashboard",
                            style={"margin": "0 0 4px", "fontSize": "26px",
                                   "fontWeight": "700", "letterSpacing": "-0.02em"},
                        ),
                        html.P(
                            "Unemployment & Inflation — Federal Reserve Economic Data (FRED)",
                            style={"margin": "0", "color": TEXT_MUTED, "fontSize": "13px"},
                        ),
                    ]
                ),
                html.Div(
                    html.Span(
                        "LIVE DATA",
                        style={
                            "backgroundColor": "#064e3b",
                            "color": "#6ee7b7",
                            "padding": "4px 10px",
                            "borderRadius": "99px",
                            "fontSize": "11px",
                            "fontWeight": "600",
                            "letterSpacing": "0.08em",
                        },
                    )
                ),
            ],
            style={"display": "flex", "justifyContent": "space-between",
                   "alignItems": "center", "marginBottom": "28px"},
        ),

        # KPI strip
        build_kpi_row(df),

        # Tabs
        dcc.Tabs(
            id="tabs",
            value="time-series",
            style={"marginBottom": "0"},
            children=[
                dcc.Tab(label="Time Series", value="time-series",
                        style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Phillips Curve", value="phillips",
                        style=tab_style, selected_style=tab_selected_style),
            ],
        ),
        html.Div(
            id="tab-content",
            style={
                "border": f"1px solid {BORDER}",
                "borderTop": "none",
                "borderRadius": "0 6px 6px 6px",
                "backgroundColor": BACKGROUND,
                "padding": "8px",
            },
        ),

        # Footer
        html.P(
            f"Source: U.S. Bureau of Labor Statistics & Bureau of Economic Analysis via FRED  ·  "
            f"Data through {df.index[-1].strftime('%B %Y')}",
            style={"color": TEXT_MUTED, "fontSize": "11px",
                   "textAlign": "center", "marginTop": "20px"},
        ),
    ],
)


@app.callback(
    dash.Output("tab-content", "children"),
    dash.Input("tabs", "value"),
)
def render_tab(tab: str):
    if tab == "time-series":
        fig = build_time_series(df)
    else:
        fig = build_phillips_curve(df)

    return dcc.Graph(
        figure=fig,
        config={"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {"format": "png", "scale": 2}},
        style={"height": "560px"},
    )


if __name__ == "__main__":
    print("Dashboard running at http://127.0.0.1:8050", flush=True)
    app.run(debug=False, host="127.0.0.1", port=8050)
