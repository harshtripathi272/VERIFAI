"""
Interactive Evidence Report Generator

Generates self-contained HTML evidence reports for VERIFAI diagnostic outputs.
Each report includes:
- Diagnosis header with confidence meter
- CheXbert structured pathology labels
- Clinical history from FHIR (Historian)
- Literature citations from PubMed
- Debate transcript
- Uncertainty trajectory visualization (SVG)
- Safety guardrails check results
- Full audit trail
"""

import json
import html as html_lib
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

# Import safety guardrails - will use it if available
try:
    from safety.guardrails import run_safety_check
except ImportError:
    run_safety_check = None


def _escape(text: str) -> str:
    """HTML-escape user content."""
    return html_lib.escape(str(text)) if text else ""


def _confidence_color(conf: float) -> str:
    """Return color based on confidence level."""
    if conf >= 0.8:
        return "#22c55e"  # green
    elif conf >= 0.6:
        return "#eab308"  # yellow
    elif conf >= 0.4:
        return "#f97316"  # orange
    else:
        return "#ef4444"  # red


def _safety_color(score: float) -> str:
    """Return color based on safety score."""
    if score >= 0.8:
        return "#22c55e"
    elif score >= 0.6:
        return "#eab308"
    elif score >= 0.4:
        return "#f97316"
    else:
        return "#ef4444"


def _build_uncertainty_svg(trajectory: list) -> str:
    """Generate SVG uncertainty trajectory chart."""
    if not trajectory:
        return '<p style="color: #666; font-style: italic;">No trajectory data available</p>'

    width = 600
    height = 200
    padding = 40
    plot_w = width - 2 * padding
    plot_h = height - 2 * padding

    n = len(trajectory)
    if n < 2:
        return '<p style="color: #666; font-style: italic;">Insufficient data points for trajectory</p>'

    # Build SVG
    svg = f'<svg viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px;height:auto;">'

    # Background
    svg += f'<rect x="0" y="0" width="{width}" height="{height}" rx="12" fill="#0a0a0f" />'

    # Grid lines
    for i in range(5):
        y = padding + (plot_h / 4) * i
        svg += f'<line x1="{padding}" y1="{y}" x2="{width-padding}" y2="{y}" stroke="#1a1a2e" stroke-width="1" />'
        label = f"{1.0 - i * 0.25:.0%}"
        svg += f'<text x="{padding-8}" y="{y+4}" text-anchor="end" fill="#555" font-size="10" font-family="monospace">{label}</text>'

    # Compute points
    points = []
    for i, entry in enumerate(trajectory):
        x = padding + (plot_w / (n - 1)) * i
        val = entry.get("after", entry.get("uncertainty", 0.5))
        y = padding + plot_h * (1.0 - val)
        points.append((x, y, entry))

    # Gradient fill under the line
    path_data = f"M{points[0][0]},{points[0][1]} "
    for x, y, _ in points[1:]:
        path_data += f"L{x},{y} "
    path_data += f"L{points[-1][0]},{padding + plot_h} L{points[0][0]},{padding + plot_h} Z"

    svg += f'<defs><linearGradient id="ug" x1="0%" y1="0%" x2="0%" y2="100%">'
    svg += '<stop offset="0%" stop-color="#00E5FF" stop-opacity="0.3" />'
    svg += '<stop offset="100%" stop-color="#00E5FF" stop-opacity="0.02" />'
    svg += '</linearGradient></defs>'
    svg += f'<path d="{path_data}" fill="url(#ug)" />'

    # Line
    line_data = f"M{points[0][0]},{points[0][1]} "
    for x, y, _ in points[1:]:
        line_data += f"L{x},{y} "
    svg += f'<path d="{line_data}" fill="none" stroke="#00E5FF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />'

    # Points and labels
    for x, y, entry in points:
        agent = entry.get("agent", "?")
        val = entry.get("after", entry.get("uncertainty", 0.5))
        ig = entry.get("ig", 0)

        color = "#00E5FF" if ig >= 0 else "#ef4444"
        svg += f'<circle cx="{x}" cy="{y}" r="5" fill="{color}" stroke="#0a0a0f" stroke-width="2" />'
        svg += f'<text x="{x}" y="{padding + plot_h + 18}" text-anchor="middle" fill="#888" font-size="9" font-family="monospace">{_escape(agent[:6])}</text>'

    svg += '</svg>'
    return svg


def generate_evidence_report(state: dict, session_id: str = "") -> str:
    """
    Generate a self-contained HTML evidence report from VERIFAI workflow state.

    Args:
        state: Complete VerifaiState dict
        session_id: Session identifier

    Returns:
        Complete HTML string (self-contained, no external dependencies)
    """
    # Extract data
    final_dx = state.get("final_diagnosis")
    rad_output = state.get("radiologist_output")
    chexbert_output = state.get("chexbert_output")
    historian_output = state.get("historian_output")
    literature_output = state.get("literature_output")
    debate_output = state.get("debate_output")
    critic_output = state.get("critic_output")
    trace = state.get("trace", [])
    uncertainty = state.get("current_uncertainty", 0.5)
    trajectory = state.get("uncertainty_trajectory", [])

    # Diagnosis info
    diagnosis = getattr(final_dx, 'diagnosis', 'Pending') or 'Pending'
    confidence = getattr(final_dx, 'calibrated_confidence', 0.5) if final_dx else 0.5
    deferred = getattr(final_dx, 'deferred', False) if final_dx else False
    explanation = getattr(final_dx, 'explanation', '') if final_dx else ''

    # Safety check
    safety_report = None
    if run_safety_check:
        try:
            safety_report = run_safety_check(state)
        except Exception:
            pass

    conf_color = _confidence_color(confidence)
    conf_pct = int(confidence * 100)
    unc_pct = int(uncertainty * 100)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # ─── Build HTML ───
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VERIFAI Evidence Report — {_escape(session_id[:8])}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    background: #050507;
    color: #e0e0e0;
    line-height: 1.6;
    min-height: 100vh;
  }}
  .container {{ max-width: 900px; margin: 0 auto; padding: 32px 24px; }}

  /* Header */
  .header {{
    border-bottom: 1px solid #1a1a2e;
    padding-bottom: 24px;
    margin-bottom: 32px;
  }}
  .header-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 4px 12px;
    border-radius: 100px;
    border: 1px solid rgba(0, 229, 255, 0.2);
    background: rgba(0, 229, 255, 0.04);
    color: #00E5FF;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 16px;
  }}
  .header-badge .dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00E5FF;
  }}
  .header h1 {{
    font-size: 36px;
    font-weight: 700;
    background: linear-gradient(135deg, #00E5FF, #64FFDA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }}
  .header-meta {{
    display: flex;
    gap: 24px;
    font-size: 12px;
    color: #666;
  }}

  /* Metrics Row */
  .metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 32px;
  }}
  .metric-card {{
    background: rgba(255,255,255, 0.02);
    border: 1px solid rgba(255,255,255, 0.04);
    border-radius: 12px;
    padding: 16px;
  }}
  .metric-label {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #555;
    margin-bottom: 4px;
  }}
  .metric-value {{
    font-size: 28px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }}

  /* Section */
  .section {{
    margin-bottom: 28px;
    background: rgba(255,255,255, 0.015);
    border: 1px solid rgba(255,255,255, 0.04);
    border-radius: 16px;
    overflow: hidden;
  }}
  .section-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 16px 20px;
    font-size: 14px;
    font-weight: 600;
    color: #ccc;
    border-bottom: 1px solid rgba(255,255,255, 0.04);
    background: rgba(0,0,0, 0.2);
    cursor: pointer;
    user-select: none;
  }}
  .section-header:hover {{ background: rgba(255,255,255, 0.02); }}
  .section-body {{ padding: 20px; }}

  /* Safety */
  .safety-pass {{ border-left: 3px solid #22c55e; }}
  .safety-fail {{ border-left: 3px solid #ef4444; }}
  .flag {{ padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; font-size: 13px; }}
  .flag-high {{ background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.15); color: #fca5a5; }}
  .flag-medium {{ background: rgba(234, 179, 8, 0.08); border: 1px solid rgba(234, 179, 8, 0.15); color: #fde68a; }}
  .flag-low {{ background: rgba(34, 197, 94, 0.08); border: 1px solid rgba(34, 197, 94, 0.15); color: #86efac; }}

  /* CheXbert */
  .label-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; }}
  .label-chip {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 12px; border-radius: 8px; font-size: 13px;
  }}
  .label-present {{ background: rgba(0, 229, 255, 0.06); border: 1px solid rgba(0, 229, 255, 0.12); color: #67e8f9; }}
  .label-uncertain {{ background: rgba(234, 179, 8, 0.06); border: 1px solid rgba(234, 179, 8, 0.12); color: #fde68a; }}

  /* Facts */
  .fact {{ padding: 10px 14px; border-radius: 8px; margin-bottom: 6px; font-size: 13px; }}
  .fact-supporting {{ background: rgba(34, 197, 94, 0.06); border-left: 3px solid rgba(34, 197, 94, 0.4); color: #86efac; }}
  .fact-contradicting {{ background: rgba(239, 68, 68, 0.06); border-left: 3px solid rgba(239, 68, 68, 0.4); color: #fca5a5; }}

  /* Citations */
  .citation {{
    padding: 14px; border-radius: 10px; margin-bottom: 8px;
    background: rgba(255,255,255, 0.02); border: 1px solid rgba(255,255,255, 0.04);
    transition: border-color 0.2s;
  }}
  .citation:hover {{ border-color: rgba(0, 229, 255, 0.15); }}
  .citation-title {{ font-size: 14px; font-weight: 600; color: #ccc; margin-bottom: 4px; }}
  .citation-meta {{ font-size: 11px; color: #555; font-family: monospace; }}
  .citation-summary {{ font-size: 13px; color: #888; margin-top: 6px; border-left: 2px solid rgba(0, 229, 255, 0.2); padding-left: 12px; }}
  .strength-badge {{
    display: inline-block; font-size: 10px; padding: 2px 8px; border-radius: 100px;
    text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
  }}
  .strength-high {{ background: rgba(34, 197, 94, 0.1); color: #22c55e; }}
  .strength-medium {{ background: rgba(234, 179, 8, 0.1); color: #eab308; }}
  .strength-low {{ background: rgba(255, 255, 255, 0.05); color: #888; }}

  /* Debate */
  .debate-round {{
    margin-bottom: 16px; padding: 14px; border-radius: 10px;
    background: rgba(0, 0, 0, 0.2); border: 1px solid rgba(255,255,255, 0.04);
  }}
  .debate-round-header {{ font-size: 12px; font-weight: 700; color: #00E5FF; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.1em; }}
  .debate-arg {{ margin: 6px 0; padding: 8px 12px; border-radius: 6px; font-size: 13px; }}
  .arg-critic {{ background: rgba(239, 68, 68, 0.05); border-left: 2px solid #ef4444; color: #fca5a5; }}
  .arg-historian {{ background: rgba(124, 77, 255, 0.05); border-left: 2px solid #7C4DFF; color: #b794f4; }}
  .arg-literature {{ background: rgba(255, 110, 64, 0.05); border-left: 2px solid #FF6E40; color: #fdba74; }}

  /* Audit */
  .trace-list {{ list-style: none; }}
  .trace-item {{
    padding: 8px 14px; border-radius: 8px; margin-bottom: 4px;
    font-family: monospace; font-size: 12px; color: #888;
    background: rgba(0, 0, 0, 0.15); border: 1px solid rgba(255,255,255, 0.03);
  }}

  /* Footer */
  .footer {{
    margin-top: 40px; padding-top: 20px;
    border-top: 1px solid #1a1a2e;
    text-align: center; font-size: 11px; color: #444;
  }}
</style>
</head>
<body>
<div class="container">
"""

    # ─── HEADER ───
    status_label = "DEFERRED" if deferred else "FINALIZED"
    html += f"""
  <div class="header">
    <div class="header-badge"><span class="dot"></span> VERIFAI Evidence Report</div>
    <h1>{_escape(diagnosis)}</h1>
    <div class="header-meta">
      <span>Session: {_escape(session_id[:12])}</span>
      <span>Generated: {timestamp}</span>
      <span>Status: {status_label}</span>
    </div>
    {'<p style="margin-top:12px; font-size:14px; color:#888;">' + _escape(explanation) + '</p>' if explanation else ''}
  </div>
"""

    # ─── METRICS ───
    safety_score_val = safety_report.safety_score if safety_report else 1.0
    safety_col = _safety_color(safety_score_val)
    html += f"""
  <div class="metrics">
    <div class="metric-card">
      <div class="metric-label">Confidence</div>
      <div class="metric-value" style="color:{conf_color}">{conf_pct}<span style="font-size:16px">%</span></div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Uncertainty</div>
      <div class="metric-value" style="color:#00E5FF">{unc_pct}<span style="font-size:16px">%</span></div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Safety Score</div>
      <div class="metric-value" style="color:{safety_col}">{int(safety_score_val * 100)}<span style="font-size:16px">%</span></div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Debate Rounds</div>
      <div class="metric-value" style="color:#E040FB">{len(getattr(debate_output, 'rounds', [])) if debate_output else 0}</div>
    </div>
  </div>
"""

    # ─── SAFETY GUARDRAILS ───
    if safety_report:
        safety_class = "safety-pass" if safety_report.passed else "safety-fail"
        html += f"""
  <div class="section {safety_class}">
    <div class="section-header">🛡️ Safety Guardrails — {'PASSED' if safety_report.passed else 'FAILED'}</div>
    <div class="section-body">
      <p style="font-size: 13px; color: #888; margin-bottom: 12px;">{_escape(safety_report.summary)}</p>
"""
        if safety_report.critical_findings:
            html += '<p style="font-size:12px; font-weight:600; color:#ef4444; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.1em;">Critical Findings</p>'
            for cf in safety_report.critical_findings:
                html += f'<div class="flag flag-high">🔴 <strong>{_escape(cf.condition)}</strong> [{cf.urgency}] — {_escape(cf.action)}</div>'

        if safety_report.red_flags:
            html += '<p style="font-size:12px; font-weight:600; color:#eab308; margin-top:12px; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.1em;">Red Flags</p>'
            for rf in safety_report.red_flags:
                css_class = f"flag-{rf.severity}"
                html += f'<div class="flag {css_class}"><strong>{_escape(rf.flag_type)}</strong>: {_escape(rf.description)}</div>'

        if safety_report.recommendations:
            html += '<p style="font-size:12px; font-weight:600; color:#888; margin-top:12px; margin-bottom:8px;">Recommendations</p>'
            for rec in safety_report.recommendations:
                html += f'<p style="font-size:13px; color:#aaa; margin-bottom:4px;">• {_escape(rec)}</p>'

        html += """
    </div>
  </div>
"""

    # ─── CHEXBERT LABELS ───
    if chexbert_output and hasattr(chexbert_output, 'labels') and chexbert_output.labels:
        html += """
  <div class="section">
    <div class="section-header">🏷️ CheXbert Structured Pathology</div>
    <div class="section-body">
      <div class="label-grid">
"""
        for condition, status in chexbert_output.labels.items():
            css = "label-present" if status == "present" else "label-uncertain"
            icon = "✓" if status == "present" else "?"
            html += f'<div class="label-chip {css}"><span>{_escape(condition)}</span><span>{icon} {_escape(status)}</span></div>'
        html += """
      </div>
    </div>
  </div>
"""

    # ─── UNCERTAINTY TRAJECTORY ───
    if trajectory:
        svg_chart = _build_uncertainty_svg(trajectory)
        html += f"""
  <div class="section">
    <div class="section-header">📈 Uncertainty Cascade (MUC Trajectory)</div>
    <div class="section-body" style="text-align:center;">
      {svg_chart}
      <p style="font-size:11px; color:#555; margin-top:8px;">Each point shows system uncertainty after an agent processes the case. Green = confirmed, Red = challenged.</p>
    </div>
  </div>
"""

    # ─── RADIOLOGIST REPORT ───
    if rad_output:
        findings = getattr(rad_output, 'findings', '')
        impression = getattr(rad_output, 'impression', '')
        html += f"""
  <div class="section">
    <div class="section-header">🩺 Radiologist Report</div>
    <div class="section-body">
      <p style="font-size:12px; font-weight:600; color:#00E5FF; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.1em;">Findings</p>
      <p style="font-size:13px; color:#aaa; margin-bottom:16px; line-height:1.7;">{_escape(findings)}</p>
      <p style="font-size:12px; font-weight:600; color:#00E5FF; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.1em;">Impression</p>
      <blockquote style="font-size:14px; color:#67e8f9; font-style:italic; border-left:2px solid rgba(0,229,255,0.3); padding-left:14px; line-height:1.7;">
        &ldquo;{_escape(impression)}&rdquo;
      </blockquote>
    </div>
  </div>
"""

    # ─── CLINICAL HISTORY (HISTORIAN) ───
    if historian_output:
        supporting = getattr(historian_output, 'supporting_facts', [])
        contradicting = getattr(historian_output, 'contradicting_facts', [])
        clin_summary = getattr(historian_output, 'clinical_summary', '')

        html += f"""
  <div class="section">
    <div class="section-header">📋 Clinical History (FHIR)</div>
    <div class="section-body">
      {'<p style="font-size:13px; color:#888; margin-bottom:12px;">' + _escape(clin_summary) + '</p>' if clin_summary else ''}
"""
        if supporting:
            html += '<p style="font-size:12px; font-weight:600; color:#22c55e; margin-bottom:6px;">Supporting Evidence</p>'
            for fact in supporting:
                html += f'<div class="fact fact-supporting">{_escape(getattr(fact, "description", str(fact)))}</div>'
        if contradicting:
            html += '<p style="font-size:12px; font-weight:600; color:#ef4444; margin-top:10px; margin-bottom:6px;">Contradicting Evidence</p>'
            for fact in contradicting:
                html += f'<div class="fact fact-contradicting">{_escape(getattr(fact, "description", str(fact)))}</div>'
        html += """
    </div>
  </div>
"""

    # ─── LITERATURE CITATIONS ───
    if literature_output and hasattr(literature_output, 'citations') and literature_output.citations:
        html += """
  <div class="section">
    <div class="section-header">📚 Literature Evidence</div>
    <div class="section-body">
"""
        strength = getattr(literature_output, 'overall_evidence_strength', 'low')
        html += f'<p style="font-size:12px; color:#888; margin-bottom:12px;">Overall evidence strength: <span class="strength-badge strength-{strength}">{strength}</span></p>'

        for cite in literature_output.citations:
            title = getattr(cite, 'title', 'Untitled')
            pmid = getattr(cite, 'pmid', '')
            journal = getattr(cite, 'journal', '')
            year = getattr(cite, 'year', '')
            summary = getattr(cite, 'relevance_summary', '')
            ev_str = getattr(cite, 'evidence_strength', 'low')

            html += f"""
      <div class="citation">
        <div style="display:flex; justify-content:space-between; align-items:start;">
          <div class="citation-title">{_escape(title)}</div>
          <span class="strength-badge strength-{ev_str}">{ev_str}</span>
        </div>
        <div class="citation-meta">{_escape(journal)} • {year} • PMID: {_escape(pmid)}</div>
        {'<div class="citation-summary">' + _escape(summary) + '</div>' if summary else ''}
      </div>
"""
        html += """
    </div>
  </div>
"""

    # ─── DEBATE TRANSCRIPT ───
    if debate_output and hasattr(debate_output, 'rounds') and debate_output.rounds:
        consensus = getattr(debate_output, 'final_consensus', False)
        html += f"""
  <div class="section">
    <div class="section-header">⚖️ Debate Transcript — {'Consensus Reached' if consensus else 'No Consensus'}</div>
    <div class="section-body">
"""
        for rnd in debate_output.rounds:
            rnum = getattr(rnd, 'round_number', '?')
            html += f'<div class="debate-round"><div class="debate-round-header">Round {rnum}</div>'

            critic_arg = getattr(rnd, 'critic_challenge', None)
            hist_arg = getattr(rnd, 'historian_response', None)
            lit_arg = getattr(rnd, 'literature_response', None)

            if critic_arg:
                html += f'<div class="debate-arg arg-critic"><strong>Critic:</strong> {_escape(getattr(critic_arg, "argument", ""))}</div>'
            if hist_arg:
                html += f'<div class="debate-arg arg-historian"><strong>Historian:</strong> {_escape(getattr(hist_arg, "argument", ""))}</div>'
            if lit_arg:
                html += f'<div class="debate-arg arg-literature"><strong>Literature:</strong> {_escape(getattr(lit_arg, "argument", ""))}</div>'

            delta = getattr(rnd, 'confidence_delta', 0)
            if delta:
                html += f'<p style="font-size:11px; color:#555; margin-top:6px;">Confidence Δ: {delta:+.3f}</p>'

            html += '</div>'

        summary = getattr(debate_output, 'debate_summary', '')
        if summary:
            html += f'<p style="font-size:13px; color:#888; margin-top:12px; border-left:2px solid #E040FB33; padding-left:12px;">{_escape(summary)}</p>'

        html += """
    </div>
  </div>
"""

    # ─── AUDIT TRAIL ───
    if trace:
        html += """
  <div class="section">
    <div class="section-header">🔍 Audit Trail</div>
    <div class="section-body">
      <ul class="trace-list">
"""
        for entry in trace:
            html += f'<li class="trace-item">{_escape(entry)}</li>'
        html += """
      </ul>
    </div>
  </div>
"""

    # ─── FOOTER ───
    html += f"""
  <div class="footer">
    <p>Generated by VERIFAI — Verified Evidence-Based Clinical AI</p>
    <p style="margin-top:4px;">⚠️ Research prototype. Not for clinical use without regulatory approval.</p>
    <p style="margin-top:4px;">Report generated: {timestamp} • Safety Score: {int(safety_score_val * 100)}%</p>
  </div>
</div>
</body>
</html>"""

    return html


def save_evidence_report(state: dict, session_id: str = "", output_dir: str = "output/reports") -> str:
    """
    Generate and save an evidence report to disk.

    Args:
        state: Complete VerifaiState dict
        session_id: Session identifier
        output_dir: Directory to save reports in

    Returns:
        Path to generated HTML file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    html = generate_evidence_report(state, session_id)
    filename = f"{session_id or 'report'}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
    filepath = Path(output_dir) / filename
    filepath.write_text(html, encoding="utf-8")
    return str(filepath)
