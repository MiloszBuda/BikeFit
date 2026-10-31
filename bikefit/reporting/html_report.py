"""Standalone HTML report generator with embedded base64 graphics and responsive design."""

import base64
from pathlib import Path
from typing import Optional

from bikefit.rules.engine import FittingReport, RecommendationCategory


def generate_html_report(
    report: FittingReport,
    chart_image_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
) -> str:
    """
    Generate a self-contained, responsive HTML bike fitting report.
    Embeds the chart image as base64 for zero-dependency viewing and sharing.
    """
    stats = report.cycle_stats

    # Embed chart if provided
    chart_base64 = ""
    if chart_image_path and Path(chart_image_path).exists():
        with open(chart_image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
            chart_base64 = f"data:image/png;base64,{b64_data}"

    def status_badge(val: float, min_v: float, max_v: float) -> str:
        if min_v <= val <= max_v:
            return '<span class="badge badge-success">✅ W normie</span>'
        diff = min(abs(val - min_v), abs(val - max_v))
        if diff <= 4.0:
            return '<span class="badge badge-warning">⚠️ Wymaga korekty</span>'
        return '<span class="badge badge-danger">❌ Poza zakresem</span>'

    table_rows = [
        ("Zgięcie kolana w DMP", f"{stats.knee_flexion_bdc_median:.1f}°", "25° – 35°", status_badge(stats.knee_flexion_bdc_median, 25.0, 35.0)),
        ("Kąt wewnętrzny kolana w DMP", f"{stats.knee_angle_bdc_median:.1f}°", "145° – 155°", status_badge(stats.knee_angle_bdc_median, 145.0, 155.0)),
        ("Kąt biodra w GMP", f"{stats.hip_angle_tdc_median:.1f}°", "≥ 45°", '<span class="badge badge-success">✅ W normie</span>' if stats.hip_angle_tdc_median >= 45 else '<span class="badge badge-danger">❌ Za mocno zamknięte</span>'),
        ("Pochylenie tułowia", f"{stats.torso_angle_median:.1f}°", "40° – 50°", '<span class="badge badge-info">ℹ️ Rekreacja/Endurance</span>' if 40 <= stats.torso_angle_median <= 50 else '<span class="badge badge-warning">ℹ️ Pozycja niestandardowa</span>'),
        ("Kąt ugięcia łokcia", f"{stats.elbow_angle_median:.1f}°", "150° – 165°", status_badge(stats.elbow_angle_median, 150.0, 165.0)),
        ("Nachylenie stopy w DMP", f"{stats.foot_angle_bdc_median:.1f}°", "-5° do +15°", status_badge(stats.foot_angle_bdc_median, -5.0, 15.0)),
    ]

    table_html = "\n".join(
        f"<tr><td><strong>{name}</strong></td><td class='text-center'>{val}</td><td class='text-center'>{target}</td><td class='text-center'>{badge}</td></tr>"
        for name, val, target, badge in table_rows
    )

    cards_html = []
    for i, rec in enumerate(report.recommendations, 1):
        is_saddle = rec.category in (RecommendationCategory.SADDLE_HEIGHT, RecommendationCategory.SADDLE_FORE_AFT)
        card_class = "card-primary" if is_saddle else "card-secondary"
        conf_badge = f'<span class="badge-pill badge-{rec.confidence.name.lower()}">Pewność: {rec.confidence.value}</span>'

        card = f"""
        <div class="rec-card {card_class}">
            <div class="rec-header">
                <div class="rec-title">
                    <span class="rec-number">#{i}</span>
                    <span class="rec-cat">{rec.category.value}</span>
                </div>
                {conf_badge}
            </div>
            <div class="rec-action">
                👉 <strong>{rec.action_summary}</strong>
            </div>
            <div class="rec-meta">
                <span>Wartość zmierzona: <strong>{rec.measured_value}{rec.unit}</strong> (zakres docelowy: {rec.target_range[0]}–{rec.target_range[1]}{rec.unit})</span>
            </div>
            <div class="rec-body">
                <p><strong>Uzasadnienie biomechaniczne:</strong> {rec.rationale}</p>
                <p class="rec-alt"><strong>Wskazówka fittera:</strong> {rec.alternatives}</p>
            </div>
        </div>
        """
        cards_html.append(card)

    chart_html = ""
    if chart_base64:
        chart_html = f"""
        <div class="section-card">
            <h2>📈 Wykresy kinematyki i cykli pedałowania</h2>
            <img src="{chart_base64}" alt="Wykres kinematyki bike fittingu" class="chart-img"/>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raport Bike Fittingu - BikeFit AI</title>
    <style>
        :root {{
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #0f172a;
            --text-muted: #64748b;
            --primary: #0284c7;
            --primary-dark: #0369a1;
            --success: #16a34a;
            --warning: #d97706;
            --danger: #dc2626;
            --border: #e2e8f0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white;
            padding: 32px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 28px; font-weight: 800; }}
        .header p {{ margin: 0; color: #94a3b8; font-size: 15px; }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 24px;
        }}
        .stat-box {{
            background: rgba(255,255,255,0.08);
            padding: 16px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-box .label {{ font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
        .stat-box .value {{ font-size: 22px; font-weight: 700; color: #38bdf8; margin-top: 4px; }}
        .section-card {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 28px;
            margin-bottom: 24px;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .section-card h2 {{
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 20px;
            font-weight: 700;
            border-bottom: 2px solid var(--border);
            padding-bottom: 12px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 8px;
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ background: #f1f5f9; font-weight: 600; color: var(--text-muted); font-size: 13px; }}
        .text-center {{ text-align: center; }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-success {{ background: #dcfce7; color: #15803d; }}
        .badge-warning {{ background: #fef3c7; color: #b45309; }}
        .badge-danger {{ background: #fee2e2; color: #b91c1c; }}
        .badge-info {{ background: #e0f2fe; color: #0369a1; }}
        .badge-pill {{
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-high {{ background: #dcfce7; color: #166534; }}
        .badge-medium {{ background: #fef3c7; color: #92400e; }}
        .badge-low {{ background: #f1f5f9; color: #475569; }}
        .rec-card {{
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            border-left: 6px solid var(--primary);
            background: #ffffff;
            border: 1px solid var(--border);
            border-left-width: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }}
        .card-primary {{ border-left-color: #0284c7; background: #f0f9ff; }}
        .card-secondary {{ border-left-color: #8b5cf6; background: #faf5ff; }}
        .rec-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .rec-title {{ font-size: 15px; font-weight: 700; }}
        .rec-number {{ color: var(--primary); margin-right: 6px; }}
        .rec-action {{
            font-size: 17px;
            color: #0f172a;
            margin-bottom: 8px;
        }}
        .rec-meta {{
            font-size: 13px;
            color: var(--text-muted);
            margin-bottom: 12px;
        }}
        .rec-body p {{ margin: 6px 0; font-size: 14px; }}
        .rec-alt {{ color: #475569; font-style: italic; }}
        .chart-img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        .disclaimer {{
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 12px;
            padding: 16px 20px;
            font-size: 13px;
            color: #92400e;
            margin-top: 32px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚴 BikeFit AI - Raport analizy pozycji</h1>
            <p>Automatyczna analiza biomechaniki jazdy na trenażerze (RTMPose Halpe26 + Segmentacja cykli)</p>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="label">Strona analizowana</div>
                    <div class="value">{stats.side.upper()}</div>
                </div>
                <div class="stat-box">
                    <div class="label">Średnia kadencja</div>
                    <div class="value">{stats.cadence_rpm_median:.0f} RPM</div>
                </div>
                <div class="stat-box">
                    <div class="label">Zbadane obroty korby</div>
                    <div class="value">{len(stats.retained_cycles)} cykli</div>
                </div>
                <div class="stat-box">
                    <div class="label">Zgięcie kolana w DMP</div>
                    <div class="value">{stats.knee_flexion_bdc_median:.1f}°</div>
                </div>
            </div>
        </div>

        <div class="section-card">
            <h2>📊 Tabela zmierzonych parametrów stawowych</h2>
            <table>
                <thead>
                    <tr>
                        <th>Parametr stawu</th>
                        <th class="text-center">Zmierzona mediana</th>
                        <th class="text-center">Zakres docelowy</th>
                        <th class="text-center">Status</th>
                    </tr>
                </thead>
                <tbody>
                    {table_html}
                </tbody>
            </table>
        </div>

        {chart_html}

        <div class="section-card">
            <h2>💡 Rekomendacje zmian ustawienia roweru</h2>
            {"".join(cards_html)}
        </div>

        <div class="disclaimer">
            <strong>⚠️ Zastrzeżenie medyczne:</strong> {report.medical_disclaimer}
        </div>
    </div>
</body>
</html>
"""
    if output_path:
        out_f = Path(output_path)
        out_f.parent.mkdir(parents=True, exist_ok=True)
        with open(out_f, "w", encoding="utf-8") as f:
            f.write(html)

    return html
