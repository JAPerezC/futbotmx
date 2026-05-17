"""Dashboard HTML interactivo con plotly.

Cumple § 3.5.2 de la convocatoria: "Dashboards o anotaciones que narren
la historia del partido". Embed standalone (un solo HTML autocontenido).
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

from src.utils.calib import FIELD_LENGTH_MM, FIELD_WIDTH_MM


EVENT_COLOR = {
    "kick": "#FFD400",
    "goal": "#00C853",
    "retention": "#D32F2F",
    "no_progress": "#FF8F00",
    "damaged": "#1976D2",
    "pass": "#7B1FA2",
    "interception": "#E040FB",
    "collision": "#F44336",
}

TEAM_COLOR_HEX = {"A": "#9B27B0", "B": "#ECEFF1", None: "#9E9E9E"}


def _event_timeline(events: list[dict]) -> go.Figure:
    fig = go.Figure()
    if not events:
        fig.add_annotation(
            text="Sin eventos detectados",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig
    types = sorted({e["type"] for e in events})
    y_pos = {t: i for i, t in enumerate(types)}
    for ev in events:
        c = EVENT_COLOR.get(ev["type"], "#888888")
        fig.add_trace(
            go.Scatter(
                x=[ev["t"]],
                y=[y_pos[ev["type"]]],
                mode="markers",
                marker=dict(
                    color=c,
                    size=12,
                    symbol="diamond",
                    line=dict(color="white", width=1),
                ),
                name=ev["type"],
                showlegend=False,
                hovertemplate=(
                    f"<b>{ev['type'].upper()}</b><br>t = {ev['t']:.2f} s<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Eventos detectados (timeline)",
        xaxis_title="Tiempo (s)",
        yaxis=dict(
            tickmode="array", tickvals=list(y_pos.values()), ticktext=list(y_pos.keys())
        ),
        height=320,
        margin=dict(l=80, r=20, t=40, b=40),
        plot_bgcolor="#1A1A1A",
        paper_bgcolor="#1A1A1A",
        font=dict(color="#EEEEEE"),
    )
    return fig


def _possession_chart(stats: dict) -> go.Figure:
    pa = stats["possession_pct"]["A"]
    pb = stats["possession_pct"]["B"]
    fig = go.Figure(
        go.Bar(
            x=[pa, pb],
            y=["Equipo A", "Equipo B"],
            orientation="h",
            text=[f"{pa:.1f} %", f"{pb:.1f} %"],
            marker_color=[TEAM_COLOR_HEX["A"], TEAM_COLOR_HEX["B"]],
        )
    )
    fig.update_layout(
        title="Posesión del balón (% del tiempo)",
        xaxis_range=[0, 100],
        height=240,
        plot_bgcolor="#1A1A1A",
        paper_bgcolor="#1A1A1A",
        font=dict(color="#EEEEEE"),
    )
    return fig


def _topdown_trails(
    robot_positions: dict[str, list], ball_positions: list
) -> go.Figure:
    import numpy as np

    fig = go.Figure()
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=FIELD_LENGTH_MM,
        y1=FIELD_WIDTH_MM,
        line=dict(color="white", width=2),
    )
    fig.add_shape(
        type="line",
        x0=FIELD_LENGTH_MM / 2,
        y0=0,
        x1=FIELD_LENGTH_MM / 2,
        y1=FIELD_WIDTH_MM,
        line=dict(color="white", width=1),
    )
    fig.add_shape(
        type="circle",
        x0=FIELD_LENGTH_MM / 2 - 300,
        y0=FIELD_WIDTH_MM / 2 - 300,
        x1=FIELD_LENGTH_MM / 2 + 300,
        y1=FIELD_WIDTH_MM / 2 + 300,
        line=dict(color="white", width=1),
    )
    fig.add_shape(
        type="rect",
        x0=-50,
        y0=FIELD_WIDTH_MM / 2 - 300,
        x1=0,
        y1=FIELD_WIDTH_MM / 2 + 300,
        fillcolor="yellow",
        line_width=0,
    )
    fig.add_shape(
        type="rect",
        x0=FIELD_LENGTH_MM,
        y0=FIELD_WIDTH_MM / 2 - 300,
        x1=FIELD_LENGTH_MM + 50,
        y1=FIELD_WIDTH_MM / 2 + 300,
        fillcolor="blue",
        line_width=0,
    )
    for tid, positions in robot_positions.items():
        if len(positions) < 2:
            continue
        arr = np.asarray(positions)
        fig.add_trace(
            go.Scatter(
                x=arr[:, 0],
                y=arr[:, 1],
                mode="lines+markers",
                name=f"track {tid}",
                line=dict(width=2),
                marker=dict(size=4),
            )
        )
    if ball_positions:
        arr = np.asarray(ball_positions)
        fig.add_trace(
            go.Scatter(
                x=arr[:, 0],
                y=arr[:, 1],
                mode="lines",
                name="balón",
                line=dict(color="orange", width=3),
            )
        )
    fig.update_layout(
        title="Trayectorias top-down (mm)",
        xaxis=dict(range=[-100, FIELD_LENGTH_MM + 100], scaleanchor="y"),
        yaxis=dict(range=[FIELD_WIDTH_MM + 100, -100]),
        height=520,
        plot_bgcolor="#1A4A2C",
        paper_bgcolor="#1A1A1A",
        font=dict(color="#EEEEEE"),
    )
    return fig


def render_dashboard(
    summary: dict,
    events: list[dict],
    tracks_record: list[dict],
    output_path: Path,
    video_name: str = "partido",
) -> Path:
    """Genera un HTML standalone con todas las visualizaciones."""
    from collections import defaultdict

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    robot_positions: dict[str, list] = defaultdict(list)
    ball_positions: list = []
    for fr in tracks_record:
        for r in fr.get("robots", []):
            tid = r.get("track_id")
            world = r.get("centroid_mm")
            if tid is not None and world is not None:
                robot_positions[str(tid)].append(world)
        b = fr.get("ball", {})
        if b and b.get("world_mm"):
            ball_positions.append(b["world_mm"])

    fig_timeline = _event_timeline(events)
    fig_possession = _possession_chart(summary)
    fig_topdown = _topdown_trails(robot_positions, ball_positions)

    score_a = summary.get("score", {}).get("A", 0)
    score_b = summary.get("score", {}).get("B", 0)

    metrics_html = f"""
    <div style='display:flex;gap:20px;justify-content:center;margin:20px 0;flex-wrap:wrap;'>
      <div style='background:#9B27B0;padding:20px 40px;border-radius:12px;text-align:center;'>
        <div style='font-size:14px;opacity:0.8;'>EQUIPO A</div>
        <div style='font-size:48px;font-weight:bold;'>{score_a}</div>
      </div>
      <div style='background:#444;padding:20px 30px;border-radius:12px;text-align:center;'>
        <div style='font-size:14px;opacity:0.8;'>POSESIÓN / ESTADÍSTICAS</div>
        <div style='font-size:22px;'>{summary["possession_pct"]["A"]:.1f}% A / {summary["possession_pct"]["B"]:.1f}% B</div>
        <div style='font-size:13px;opacity:0.7;margin-top:6px;'>v máx balón: {summary["ball"]["max_speed_mm_s"]:.0f} mm/s
          · v prom: {summary["ball"]["avg_speed_mm_s"]:.0f} mm/s</div>
        <div style='font-size:13px;opacity:0.7;'>tracks únicos: {summary["tracks_seen"]}
          · eventos totales: {sum(summary["events_by_type"].values())}</div>
      </div>
      <div style='background:#ECEFF1;color:#1A1A1A;padding:20px 40px;border-radius:12px;text-align:center;'>
        <div style='font-size:14px;opacity:0.8;'>EQUIPO B</div>
        <div style='font-size:48px;font-weight:bold;'>{score_b}</div>
      </div>
    </div>
    """

    event_grid = "<div style='display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin:10px 0;'>"
    for tname, count in sorted(summary["events_by_type"].items()):
        c = EVENT_COLOR.get(tname, "#888")
        event_grid += (
            f"<div style='background:#222;padding:8px 16px;border-radius:8px;border:2px solid {c};'>"
            f"<span style='color:{c};font-weight:bold;'>{tname.upper()}</span>: {count}</div>"
        )
    event_grid += "</div>"

    html = f"""<!doctype html><html><head>
<meta charset='utf-8'>
<title>Dashboard FutBotMX — {video_name}</title>
<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
<style>
  body {{ margin:0; padding:24px; background:#0F0F0F; color:#EEE;
          font-family:-apple-system,Segoe UI,Roboto,sans-serif; }}
  h1 {{ margin:0 0 8px; }}
  h2 {{ margin:24px 0 8px; color:#FFD400; }}
  .row {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  @media (max-width:900px) {{ .row {{ grid-template-columns:1fr; }} }}
</style>
</head><body>
<h1>futbotmx · análisis de partido</h1>
<div style='opacity:0.7;'>Video: <code>{video_name}</code> — pipeline SAM 3.1 + OC-SORT + Kalman + rule-based events</div>
{metrics_html}
{event_grid}
<h2>Trayectorias top-down</h2>
<div id='topdown'></div>
<div class='row'>
  <div><h2>Eventos en el tiempo</h2><div id='timeline'></div></div>
  <div><h2>Posesión</h2><div id='possession'></div></div>
</div>
<script>
  Plotly.newPlot('topdown',  {json.dumps(fig_topdown.to_plotly_json()["data"])},
                             {json.dumps(fig_topdown.to_plotly_json()["layout"])}, {{responsive:true}});
  Plotly.newPlot('timeline', {json.dumps(fig_timeline.to_plotly_json()["data"])},
                             {json.dumps(fig_timeline.to_plotly_json()["layout"])}, {{responsive:true}});
  Plotly.newPlot('possession', {json.dumps(fig_possession.to_plotly_json()["data"])},
                              {json.dumps(fig_possession.to_plotly_json()["layout"])}, {{responsive:true}});
</script>
</body></html>"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
