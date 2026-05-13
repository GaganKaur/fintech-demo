import sys
import traceback

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

BACKGROUND = "#0f172a"
SURFACE = "#1e293b"
BORDER = "#334155"
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


def build_phillips_curve(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

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
                marker=dict(
                    color=color,
                    size=5,
                    opacity=0.75,
                    line=dict(width=0),
                ),
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    "Unemployment: %{x:.1f}%<br>"
                    "Inflation: %{y:.1f}%<extra></extra>"
                ),
                customdata=subset.index.strftime("%b %Y"),
            )
        )

    fig.add_hline(
        y=0,
        line=dict(color=BORDER, width=1, dash="dot"),
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="Phillips Curve: Unemployment vs. Inflation by Decade",
            font=dict(size=18, color=TEXT_PRIMARY),
            x=0.03,
        ),
        xaxis=dict(**AXIS_STYLE, title="Unemployment Rate (%)"),
        yaxis=dict(**AXIS_STYLE, title="CPI Inflation YoY (%)"),
        legend=dict(**LEGEND_STYLE, title=dict(text="Decade", font=dict(color=TEXT_MUTED))),
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
