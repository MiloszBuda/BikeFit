"""Streamlit Web Application for AI-powered Bike Fitting."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from bikefit.calibration.crank import CrankCalibration
from bikefit.io.plotting import plot_kinematics_summary
from bikefit.io.storage import save_keypoints
from bikefit.io.visualization import VideoVisualizer
from bikefit.kinematics.angles import extract_kinematics
from bikefit.kinematics.scale import calibrate_scale_from_crank
from bikefit.models.rtmpose import RTMPoseEstimator
from bikefit.pipeline.video import VideoProcessor
from bikefit.reporting.html_report import generate_html_report
from bikefit.reporting.summary import format_markdown_summary, format_text_summary, save_reports
from bikefit.rules.config import FittingConfig
from bikefit.rules.engine import ConfidenceLevel, FittingRuleEngine, RecommendationCategory
from bikefit.segmentation.cycles import CycleDetector


def run_app() -> None:
    """Entrypoint to launch the Streamlit app from CLI."""
    app_path = Path(__file__).resolve()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


def main() -> None:
    st.set_page_config(
        page_title="BikeFit AI - Analiza pozycji kolarza",
        page_icon="🚴",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🚴 BikeFit AI - Analiza Pozycji na Rowerze")
    st.markdown(
        "Automatyczna analiza wideo jazdy na trenażerze: **śledzenie 26 stawów (RTMPose Halpe26)**, "
        "filtracja zerofazowa, segmentacja cykli pedałowania oraz **porady fittingowe z przeliczeniem na milimetry**."
    )

    # -------------------------------------------------------------
    # SIDEBAR: PARAMETRY ANALIZY
    # -------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Ustawienia i parametry")

        st.subheader("1. Rower i geometria")
        crank_length = st.number_input(
            "Długość ramienia korby [mm]:",
            min_value=150.0,
            max_value=185.0,
            value=172.5,
            step=2.5,
            help="Standardowe długości to np. 165 mm, 170 mm, 172.5 mm, 175 mm.",
        )

        side_choice = st.selectbox(
            "Strona kolarza skierowana do kamery:",
            options=["auto", "right", "left"],
            index=0,
            format_func=lambda x: "Automatyczna detekcja" if x == "auto" else ("Prawa (napędowa)" if x == "right" else "Lewa"),
        )

        st.subheader("2. Model RTMPose")
        model_mode = st.selectbox(
            "Tryb modelu:",
            options=["balanced", "lightweight", "performance"],
            index=0,
            help="Balanced: optymalny kompromis szybkości i precyzji. Lightweight: najszybszy.",
        )

        device_choice = st.selectbox(
            "Urządzenie obliczeniowe:",
            options=["cpu", "cuda"],
            index=0,
        )

        kpt_threshold = st.slider(
            "Próg pewności stawu (confidence):",
            min_value=0.20,
            max_value=0.70,
            value=0.35,
            step=0.05,
        )

        filter_cutoff = st.slider(
            "Filtr Butterwortha (odcięcie Hz):",
            min_value=3.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            help="Częstotliwość odcięcia szumu drżenia detektora (5 Hz jest optymalne dla kadencji 60-120 RPM).",
        )

    # -------------------------------------------------------------
    # GŁÓWNY PANEL: KROK 1 - WGRANIE WIDEO
    # -------------------------------------------------------------
    uploaded_file = st.file_uploader(
        "📁 Wgraj nagranie wideo (MP4 / MOV / AVI z boku roweru):",
        type=["mp4", "mov", "avi"],
    )

    if not uploaded_file:
        st.info("👆 Wgraj film z nagraniem jazdy na trenażerze, aby rozpocząć analizę.")
        st.markdown(
            """
            > **Wskazówki do nagrania:**
            > - Kamera ustawiona dokładnie z boku, prostopadle do osi roweru, na wysokości suportu.
            > - Stabilny statyw (brak drgań kamery).
            > - Obcisły strój kolarski (ułatwia precyzyjne śledzenie kolan, bioder i kostek).
            > - Nagranie min. 10–20 sekund ciągłego pedałowania w stałym tempie.
            """
        )
        return

    # Zapisz wgrany plik tymczasowo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(uploaded_file.read())
        video_input_path = Path(tmp_in.name)

    cap = cv2.VideoCapture(str(video_input_path))
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        st.error("Nie udało się odczytać klatek z przesłanego pliku wideo.")
        return

    h, w, _ = first_frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # -------------------------------------------------------------
    # KROK 2: KALIBRACJA OSI KORBY (SUPORTU)
    # -------------------------------------------------------------
    st.subheader("🎯 Krok 2: Wskaż środek osi korby (suport / BB)")
    col_prev, col_coords = st.columns([2, 1])

    with col_coords:
        st.markdown("Dostosuj współrzędne osi korby na pierwszej klatce:")
        default_x = int(w * 0.45)
        default_y = int(h * 0.65)

        crank_x = st.slider("Pozycja X suportu [px]:", 0, w, default_x, 1)
        crank_y = st.slider("Pozycja Y suportu [px]:", 0, h, default_y, 1)

        st.caption(f"Wymiary wideo: {w}x{h} px | {fps:.1f} FPS")

    with col_prev:
        # Rysuj celownik na pierwszej klatce
        preview_img = first_frame.copy()
        cv2.circle(preview_img, (crank_x, crank_y), 16, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(preview_img, (crank_x, crank_y), 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.line(preview_img, (crank_x - 30, crank_y), (crank_x + 30, crank_y), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(preview_img, (crank_x, crank_y - 30), (crank_x, crank_y + 30), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(preview_img, "OS KORBY (BB)", (crank_x + 15, crank_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Konwersja BGR -> RGB do wyświetlenia w Streamlit
        preview_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
        st.image(preview_rgb, caption="Podgląd pierwszej klatki z celownikiem suportu", use_container_width=True)

    # -------------------------------------------------------------
    # KROK 3: URUCHOMIENIE ANALIZY
    # -------------------------------------------------------------
    st.markdown("---")
    if st.button("🚀 Uruchom pełną analizę Bike Fittingu", type="primary"):
        with st.spinner("Inicjalizacja modelu RTMPose i przetwarzanie wideo..."):
            progress_bar = st.progress(0, text="Rozpoczynanie detekcji stawów...")

            # Inicjalizacja komponentów
            estimator = RTMPoseEstimator(mode=model_mode, device=device_choice)
            visualizer = VideoVisualizer(kpt_score_threshold=kpt_threshold)
            processor = VideoProcessor(pose_estimator=estimator, visualizer=visualizer)

            out_dir = Path(tempfile.mkdtemp(prefix="bikefit_web_"))
            out_video_path = out_dir / "annotated.mp4"

            def on_progress(current, total):
                if total > 0:
                    pct = min(100, int((current / total) * 100))
                    progress_bar.progress(pct, text=f"Analiza klatek: {current}/{total} ({pct}%)")

            # Przetwarzanie wideo
            results, metadata = processor.process_video(
                video_path=video_input_path,
                crank=(crank_x, crank_y),
                output_video_path=out_video_path,
                side_preference=side_choice if side_choice != "auto" else None,
                progress_callback=on_progress,
            )

            progress_bar.progress(100, text="Filtracja trajektorii i obliczanie kątów...")

            # Kinematyka i kąty
            kinematics = extract_kinematics(
                results=results,
                fps=metadata.fps,
                side=side_choice,
                cutoff_hz=filter_cutoff,
                score_threshold=kpt_threshold,
            )

            # Segmentacja cykli
            detector = CycleDetector()
            cycle_stats = detector.segment(kinematics, fps=metadata.fps)

            # Silnik reguł
            engine = FittingRuleEngine(config=FittingConfig())
            report = engine.evaluate(cycle_stats=cycle_stats, known_crank_length_mm=crank_length)

            # Wykres
            chart_path = out_dir / "kinematics.png"
            plot_kinematics_summary(kinematics, output_path=chart_path, cycle_stats=cycle_stats)

            # Raporty
            saved_paths = save_reports(report, output_dir=out_dir, base_name="bikefit", chart_path=chart_path)
            html_content = generate_html_report(report, chart_image_path=chart_path, output_path=saved_paths["html"])

            # Zapisz wyniki w sesji Streamlit
            st.session_state["report"] = report
            st.session_state["kinematics"] = kinematics
            st.session_state["cycle_stats"] = cycle_stats
            st.session_state["out_dir"] = out_dir
            st.session_state["out_video_path"] = out_video_path
            st.session_state["chart_path"] = chart_path
            st.session_state["html_content"] = html_content
            st.session_state["results"] = results

        st.success("✅ Analiza zakończona sukcesem!")

    # -------------------------------------------------------------
    # PREZENTACJA WYNIKÓW
    # -------------------------------------------------------------
    if "report" in st.session_state:
        report: FittingReport = st.session_state["report"]
        cycle_stats = report.cycle_stats
        out_dir = st.session_state["out_dir"]
        out_video_path = st.session_state["out_video_path"]
        chart_path = st.session_state["chart_path"]
        html_content = st.session_state["html_content"]
        results = st.session_state["results"]

        st.markdown("## 📊 Wyniki i Raport Fittingowy")

        # Metryki w kafelkach
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Analizowana strona", cycle_stats.side.upper())
        col2.metric("Kadencja (mediana)", f"{cycle_stats.cadence_rpm_median:.0f} RPM", f"±{cycle_stats.cadence_rpm_std:.1f}")
        col3.metric("Zgięcie kolana w DMP", f"{cycle_stats.knee_flexion_bdc_median:.1f}°", "cel: 25°-35°")
        col4.metric("Kąt biodra w GMP", f"{cycle_stats.hip_angle_tdc_median:.1f}°", "cel: ≥ 45°")

        tab_recs, tab_video, tab_charts, tab_data = st.tabs([
            "💡 Porady i Rekomendacje",
            "🎬 Wideo ze szkieletem",
            "📈 Wykresy kinematyki",
            "💾 Pobieranie danych i raportów",
        ])

        with tab_recs:
            st.subheader("Rekomendowane zmiany w ustawieniu roweru:")
            for i, rec in enumerate(report.recommendations, 1):
                is_saddle = rec.category in (RecommendationCategory.SADDLE_HEIGHT, RecommendationCategory.SADDLE_FORE_AFT)
                icon = "🔧" if is_saddle else "⚙️"

                with st.expander(f"{icon} #{i} {rec.category.value}: {rec.action_summary}", expanded=True):
                    st.markdown(f"**Zmierzona wartość:** `{rec.measured_value}{rec.unit}` (zakres docelowy: `{rec.target_range[0]}`–`{rec.target_range[1]}{rec.unit}`)")
                    st.markdown(f"**Poziom pewności:** `{rec.confidence.value}`")
                    st.info(f"**Uzasadnienie:** {rec.rationale}")
                    st.caption(f"**Wskazówka fittera:** {rec.alternatives}")

            st.warning(f"**Zastrzeżenie:** {report.medical_disclaimer}")

        with tab_video:
            st.subheader("Nagranie wideo z nałożoną estymacją stawów i suportem:")
            if out_video_path.exists():
                st.video(str(out_video_path))

        with tab_charts:
            st.subheader("Przebiegi kątów stawów w czasie (z zaznaczonymi DMP i GMP):")
            if chart_path.exists():
                st.image(str(chart_path), use_container_width=True)

        with tab_data:
            st.subheader("Pobierz wyniki analizy:")
            down_col1, down_col2, down_col3, down_col4 = st.columns(4)

            # 1. HTML
            down_col1.download_button(
                "📄 Raport HTML (Pełny)",
                data=html_content,
                file_name="bikefit_raport.html",
                mime="text/html",
            )

            # 2. Markdown
            md_text = format_markdown_summary(report)
            down_col2.download_button(
                "📝 Raport Markdown",
                data=md_text,
                file_name="bikefit_raport.md",
                mime="text/markdown",
            )

            # 3. Kąty CSV
            kinematics: KinematicsResult = st.session_state["kinematics"]
            df_angles = kinematics.to_dataframe()
            down_col3.download_button(
                "📊 Kąty (CSV)",
                data=df_angles.to_csv(index=False),
                file_name="bikefit_angles.csv",
                mime="text/csv",
            )

            # 4. Punkty kluczowe CSV
            df_kpts = pd.DataFrame([
                {
                    "frame_idx": r.frame_idx,
                    "timestamp_s": r.timestamp_s,
                    "keypoint": kp.name,
                    "x": kp.x,
                    "y": kp.y,
                    "score": kp.score,
                }
                for r in results
                for kp in r.keypoints.values()
            ])
            down_col4.download_button(
                "📍 Punkty kluczowe (CSV)",
                data=df_kpts.to_csv(index=False),
                file_name="bikefit_keypoints.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
