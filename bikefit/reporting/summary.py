"""Report generation in text, Markdown, HTML, and JSON formats."""

import json
from pathlib import Path
from typing import Optional

from bikefit.rules.engine import FittingReport


def format_text_summary(report: FittingReport) -> str:
    """Generate clean human-readable terminal/text report."""
    stats = report.cycle_stats
    lines = [
        "=" * 65,
        "         RAPORT ANALIZY BIKE FITTINGU (BikeFit AI)",
        "=" * 65,
        f"Analizowana strona:           {stats.side.upper()}",
        f"Liczba wykrytych obrotów:     {len(stats.all_cycles)} (analizowano: {len(stats.retained_cycles)})",
        f"Kadencja (mediana):           {stats.cadence_rpm_median:.1f} RPM (+/- {stats.cadence_rpm_std:.1f})",
        "",
        "--- ZMIERZONE PARAMETRY BIOMECHANICZNE (MEDIANA +/- STD) ---",
        f"• Zgięcie kolana w DMP:       {stats.knee_flexion_bdc_median:.1f}° (+/- {stats.knee_flexion_bdc_std:.1f}°)  [cel: 25°-35°]",
        f"• Kąt wewnętrzny kolana:      {stats.knee_angle_bdc_median:.1f}° (+/- {stats.knee_angle_bdc_std:.1f}°)  [cel: 145°-155°]",
        f"• Zamknięcie biodra w GMP:    {stats.hip_angle_tdc_median:.1f}° (+/- {stats.hip_angle_tdc_std:.1f}°)  [cel: >= 45°]",
        f"• Kąt pochylenia tułowia:     {stats.torso_angle_median:.1f}° (+/- {stats.torso_angle_std:.1f}°)  [cel: 40°-50°]",
        f"• Kąt ugięcia łokcia:         {stats.elbow_angle_median:.1f}° (+/- {stats.elbow_angle_std:.1f}°)  [cel: 150°-165°]",
        f"• Nachylenie stopy w DMP:     {stats.foot_angle_bdc_median:.1f}° (+/- {stats.foot_angle_bdc_std:.1f}°)  [cel: -5° do +15°]",
        "",
        "--- REKOMENDACJE I WSKAZÓWKI ZMIAN ---",
    ]

    for i, rec in enumerate(report.recommendations, 1):
        lines.append(f"\n[{i}] {rec.category.value}:")
        lines.append(f"    Parametr:     {rec.parameter_name} = {rec.measured_value}{rec.unit} (zakres: {rec.target_range[0]}-{rec.target_range[1]}{rec.unit})")
        lines.append(f"    ZALECENIE:    >>> {rec.action_summary} <<<")
        lines.append(f"    Pewność:      {rec.confidence.value}")
        lines.append(f"    Uzasadnienie: {rec.rationale}")
        lines.append(f"    Wskazówka:    {rec.alternatives}")

    lines.extend([
        "",
        "=" * 65,
        report.medical_disclaimer,
        "=" * 65,
    ])
    return "\n".join(lines)


def format_markdown_summary(report: FittingReport) -> str:
    """Generate GitHub-style Markdown report."""
    stats = report.cycle_stats
    lines = [
        "# Raport analizy Bike Fittingu",
        "",
        f"- **Strona kolarza**: {stats.side.upper()}",
        f"- **Wykryte cykle pedałowania**: {len(stats.all_cycles)} (analizowano: {len(stats.retained_cycles)})",
        f"- **Średnia kadencja**: {stats.cadence_rpm_median:.1f} RPM (odchylenie standardowe: {stats.cadence_rpm_std:.1f})",
        "",
        "## Podsumowanie biomechaniczne",
        "",
        "| Parametr stawu | Zmierzona mediana | Zakres optymalny | Status |",
        "| :--- | :---: | :---: | :---: |",
        f"| Zgięcie kolana w DMP | **{stats.knee_flexion_bdc_median:.1f}°** | 25° – 35° | {'✅ Norma' if 25 <= stats.knee_flexion_bdc_median <= 35 else '⚠️ Poza zakresem'} |",
        f"| Kąt wewnętrzny kolana w DMP | **{stats.knee_angle_bdc_median:.1f}°** | 145° – 155° | {'✅ Norma' if 145 <= stats.knee_angle_bdc_median <= 155 else '⚠️ Poza zakresem'} |",
        f"| Kąt biodra w GMP | **{stats.hip_angle_tdc_median:.1f}°** | >= 45° | {'✅ Norma' if stats.hip_angle_tdc_median >= 45 else '⚠️ Zbyt mały'} |",
        f"| Kąt tułowia (wzgl. poziomu) | **{stats.torso_angle_median:.1f}°** | 40° – 50° | {'✅ Norma' if 40 <= stats.torso_angle_median <= 50 else 'ℹ️ Informacyjny'} |",
        f"| Kąt łokcia | **{stats.elbow_angle_median:.1f}°** | 150° – 165° | {'✅ Norma' if 150 <= stats.elbow_angle_median <= 165 else '⚠️ Poza zakresem'} |",
        f"| Kąt nachylenia stopy w DMP | **{stats.foot_angle_bdc_median:.1f}°** | -5° – 15° | {'✅ Norma' if -5 <= stats.foot_angle_bdc_median <= 15 else '⚠️ Poza zakresem'} |",
        "",
        "## Szczegółowe zalecenia zmian",
        "",
    ]

    for i, rec in enumerate(report.recommendations, 1):
        lines.extend([
            f"### {i}. {rec.category.value}: {rec.parameter_name}",
            f"- **Wartość zmierzona**: `{rec.measured_value}{rec.unit}` (docelowo: `{rec.target_range[0]}`–`{rec.target_range[1]}{rec.unit}`)",
            f"- **Kluczowa akcja**: **{rec.action_summary}**",
            f"- **Poziom pewności**: `{rec.confidence.value}`",
            f"- **Uzasadnienie biomechaniczne**: {rec.rationale}",
            f"- **Wskazówki i alternatywy**: {rec.alternatives}",
            "",
        ])

    lines.extend([
        "> [!IMPORTANT]",
        f"> {report.medical_disclaimer}",
        "",
    ])
    return "\n".join(lines)


def save_reports(
    report: FittingReport,
    output_dir: Path | str,
    base_name: str,
    chart_path: Optional[Path | str] = None,
) -> dict:
    """Save reports in TXT, Markdown, HTML, and JSON formats."""
    from bikefit.reporting.html_report import generate_html_report

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / f"{base_name}_report.txt"
    md_path = out_dir / f"{base_name}_report.md"
    html_path = out_dir / f"{base_name}_report.html"
    json_path = out_dir / f"{base_name}_report.json"

    # TXT
    txt_content = format_text_summary(report)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)

    # MD
    md_content = format_markdown_summary(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # HTML
    generate_html_report(report, chart_image_path=chart_path, output_path=html_path)

    # JSON
    json_data = {
        "side": report.cycle_stats.side,
        "cadence_rpm": {
            "median": report.cycle_stats.cadence_rpm_median,
            "std": report.cycle_stats.cadence_rpm_std,
        },
        "stats": {
            "knee_flexion_bdc": {
                "median": report.cycle_stats.knee_flexion_bdc_median,
                "std": report.cycle_stats.knee_flexion_bdc_std,
            },
            "knee_angle_bdc": {
                "median": report.cycle_stats.knee_angle_bdc_median,
                "std": report.cycle_stats.knee_angle_bdc_std,
            },
            "hip_angle_tdc": {
                "median": report.cycle_stats.hip_angle_tdc_median,
                "std": report.cycle_stats.hip_angle_tdc_std,
            },
            "torso_angle": {
                "median": report.cycle_stats.torso_angle_median,
                "std": report.cycle_stats.torso_angle_std,
            },
            "elbow_angle": {
                "median": report.cycle_stats.elbow_angle_median,
                "std": report.cycle_stats.elbow_angle_std,
            },
            "foot_angle_bdc": {
                "median": report.cycle_stats.foot_angle_bdc_median,
                "std": report.cycle_stats.foot_angle_bdc_std,
            },
        },
        "recommendations": [
            {
                "category": r.category.value,
                "parameter": r.parameter_name,
                "measured": r.measured_value,
                "target": r.target_range,
                "action": r.action_summary,
                "adjustment_mm": r.adjustment_mm,
                "confidence": r.confidence.value,
                "rationale": r.rationale,
                "alternatives": r.alternatives,
            }
            for r in report.recommendations
        ],
        "disclaimer": report.medical_disclaimer,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    return {
        "txt": txt_path,
        "md": md_path,
        "html": html_path,
        "json": json_path,
    }
