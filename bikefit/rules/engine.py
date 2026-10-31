"""Rule engine translating segmented kinematics and angles into actionable fit advice."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from bikefit.rules.config import FittingConfig
from bikefit.segmentation.cycles import CycleSegmentationResult


class ConfidenceLevel(str, Enum):
    HIGH = "Wysoka"
    MEDIUM = "Średnia"
    LOW = "Niska"


class RecommendationCategory(str, Enum):
    SADDLE_HEIGHT = "Wysokość siodełka (Podstawowe)"
    SADDLE_FORE_AFT = "Przesunięcie siodełka (Podstawowe)"
    COCKPIT_REACH_STACK = "Kokpit / Mostek (Wymiana części)"
    CRANK_LENGTH = "Długość korby (Wymiana części)"
    PEDALING_TECHNIQUE = "Technika pedałowania i stabilność"


@dataclass
class Recommendation:
    """Individual actionable fitting adjustment recommendation."""
    category: RecommendationCategory
    parameter_name: str
    measured_value: float
    target_range: Tuple[float, float]
    unit: str
    action_summary: str
    adjustment_mm: Optional[float]
    confidence: ConfidenceLevel
    rationale: str
    alternatives: str


@dataclass
class FittingReport:
    """Complete diagnostic fitting report containing advice and disclaimers."""
    recommendations: List[Recommendation]
    cycle_stats: CycleSegmentationResult
    medical_disclaimer: str = (
        "WAŻNE ZASTRZEŻENIE: Niniejszy raport ma charakter wyłącznie poglądowy i służy "
        "optymalizacji biomechanicznej na podstawie analizy wideo 2D. Nie stanowi on porady "
        "medycznej ani diagnostyki fizjoterapeutycznej. W przypadku występowania bólu (np. kolan, "
        "pleców, karku, drętwienia dłoni) skonsultuj się z profesjonalnym bike fitterem "
        "lub fizjoterapeutą sportowym."
    )


class FittingRuleEngine:
    """
    Expert rule engine applying clinical and sports bike fitting heuristics.
    """

    def __init__(self, config: Optional[FittingConfig] = None) -> None:
        self.config = config or FittingConfig()

    def evaluate(
        self,
        cycle_stats: CycleSegmentationResult,
        known_crank_length_mm: Optional[float] = None,
    ) -> FittingReport:
        """
        Evaluate cycling kinematics and generate prioritized, actionable recommendations.
        """
        recs: List[Recommendation] = []

        if len(cycle_stats.all_cycles) == 0:
            recs.append(
                Recommendation(
                    category=RecommendationCategory.PEDALING_TECHNIQUE,
                    parameter_name="Wykrywanie cykli pedałowania",
                    measured_value=0.0,
                    target_range=(3.0, 100.0),
                    unit=" cykli",
                    action_summary="Nagraj ponownie: brak wyraźnych cykli obrotu korby",
                    adjustment_mm=None,
                    confidence=ConfidenceLevel.HIGH,
                    rationale=(
                        "W nagraniu nie udało się wyodrębnić pełnych obrotów korby (DMP/GMP). "
                        "Możliwe przyczyny: wideo jest za krótkie (mniej niż 2-3 sekundy pedałowania), "
                        "kolarz nie pedałował w sposób ciągły, lub sylwetka była niewidoczna."
                    ),
                    alternatives=(
                        "Zalecenie: Nagraj 15–30 sekund ciągłej jazdy na trenażerze w stałym tempie "
                        "(kadencja 70-95 RPM), upewniając się, że cała sylwetka i rower mieszczą się w kadrze."
                    ),
                )
            )
            return FittingReport(recommendations=recs, cycle_stats=cycle_stats)

        # -------------------------------------------------------------
        # 1. SADDLE HEIGHT (Główny wskaźnik: Zgięcie kolana w DMP)
        # -------------------------------------------------------------
        knee_flx = cycle_stats.knee_flexion_bdc_median
        knee_std = cycle_stats.knee_flexion_bdc_std
        cfg_knee = self.config.knee_flexion_bdc

        # Confidence is high if measurement repeatability is good (low std)
        conf_knee = ConfidenceLevel.HIGH if knee_std < 3.5 else ConfidenceLevel.MEDIUM

        if knee_flx < cfg_knee.min_val:
            # Under 25 deg flexion -> knee too straight -> saddle too high
            delta_deg = cfg_knee.ideal_val - knee_flx
            raw_mm = delta_deg * self.config.saddle_height_mm_per_deg
            step_mm = min(raw_mm, self.config.max_single_saddle_step_mm)

            recs.append(
                Recommendation(
                    category=RecommendationCategory.SADDLE_HEIGHT,
                    parameter_name="Zgięcie kolana w DMP (Dolny Martwy Punkt)",
                    measured_value=round(knee_flx, 1),
                    target_range=(cfg_knee.min_val, cfg_knee.max_val),
                    unit="deg",
                    action_summary=f"Obniż siodełko o ok. {step_mm:.0f} mm (docelowo ok. {raw_mm:.0f} mm)",
                    adjustment_mm=-step_mm,
                    confidence=conf_knee,
                    rationale=(
                        f"Zgięcie kolana w DMP wynosi zaledwie {knee_flx:.1f}° (kąt wewnętrzny "
                        f"{180 - knee_flx:.1f}°). Noga jest zbyt wyprostowana, co zmusza miednicę "
                        "do kołysania na boki, przeciąża mięśnie kulszowo-goleniowe oraz ścięgna "
                        "w dole podkolanowym i może wywoływać ból odcinka lędźwiowego."
                    ),
                    alternatives=(
                        "Zalecenie: Zmniejszaj wysokość małymi krokami (maks. 5 mm na raz) i powtórz pomiar. "
                        "Jeśli stopa mocno opada palcami w dół, upewnij się, że nie kompensujesz tym wysokości."
                    ),
                )
            )
        elif knee_flx > cfg_knee.max_val:
            # Over 35 deg flexion -> knee too bent -> saddle too low
            delta_deg = knee_flx - cfg_knee.ideal_val
            raw_mm = delta_deg * self.config.saddle_height_mm_per_deg
            step_mm = min(raw_mm, self.config.max_single_saddle_step_mm)

            recs.append(
                Recommendation(
                    category=RecommendationCategory.SADDLE_HEIGHT,
                    parameter_name="Zgięcie kolana w DMP (Dolny Martwy Punkt)",
                    measured_value=round(knee_flx, 1),
                    target_range=(cfg_knee.min_val, cfg_knee.max_val),
                    unit="deg",
                    action_summary=f"Podnieś siodełko o ok. {step_mm:.0f} mm (docelowo ok. {raw_mm:.0f} mm)",
                    adjustment_mm=step_mm,
                    confidence=conf_knee,
                    rationale=(
                        f"Zgięcie kolana w DMP wynosi aż {knee_flx:.1f}° (kąt wewnętrzny "
                        f"{180 - knee_flx:.1f}°). Kolano pozostaje nadmiernie zgięte, co drastycznie "
                        "zwiększa siły nacisku na staw rzepkowo-udowy (częsta przyczyna bólu przedniej "
                        "części kolana) oraz obniża efektywność generowania mocy."
                    ),
                    alternatives=(
                        "Zalecenie: Podnieś siodełko o 5 mm, wykonaj jazdę testową i zweryfikuj kąt. "
                        "Uwaga: podnoszenie siodełka jednocześnie oddala kierownicę (zwiększa reach)."
                    ),
                )
            )
        else:
            recs.append(
                Recommendation(
                    category=RecommendationCategory.SADDLE_HEIGHT,
                    parameter_name="Zgięcie kolana w DMP",
                    measured_value=round(knee_flx, 1),
                    target_range=(cfg_knee.min_val, cfg_knee.max_val),
                    unit="deg",
                    action_summary="Wysokość siodełka prawidłowa (w normie)",
                    adjustment_mm=0.0,
                    confidence=conf_knee,
                    rationale=(
                        f"Kąt kolana w DMP wynosi {knee_flx:.1f}° i mieści się idealnie w "
                        f"zakresie docelowym ({cfg_knee.min_val}° - {cfg_knee.max_val}°)."
                    ),
                    alternatives="Brak konieczności zmian w wysokości siodełka.",
                )
            )

        # -------------------------------------------------------------
        # 2. CRANK LENGTH (Długość ramion korby: Kąt biodra w GMP)
        # -------------------------------------------------------------
        hip_tdc = cycle_stats.hip_angle_tdc_median
        cfg_hip = self.config.hip_angle_tdc

        if hip_tdc < cfg_hip.min_val:
            crank_str = f"{known_crank_length_mm:.1f} mm" if known_crank_length_mm else "obecnej długości"
            recs.append(
                Recommendation(
                    category=RecommendationCategory.CRANK_LENGTH,
                    parameter_name="Kąt zamknięcia biodra w GMP (Górny Martwy Punkt)",
                    measured_value=round(hip_tdc, 1),
                    target_range=(cfg_hip.min_val, cfg_hip.max_val),
                    unit="deg",
                    action_summary="Rozważ skrócenie ramion korby (np. o 5-7.5 mm)",
                    adjustment_mm=None,
                    confidence=ConfidenceLevel.MEDIUM,
                    rationale=(
                        f"Kąt biodra w górnej fazie obrotu (godzina 12:00) wynosi tylko {hip_tdc:.1f}° "
                        f"(poniżej zalecanych {cfg_hip.min_val}°). Zbyt mocne zamknięcie biodra "
                        "powoduje ucisk na jamę brzuszną i pachwiny, utrudnia swobodne oddychanie "
                        "oraz wymusza kołysanie miednicą w celu minięcia martwego punktu."
                    ),
                    alternatives=(
                        f"Jeśli używasz korby {crank_str}, zmiana na krótszą (np. 165 mm zamiast 172.5/175 mm) "
                        "otworzy kąt biodra o około 3-5° i obniży kolano na szczycie obrotu. "
                        "Alternatywnie: lekkie podniesienie kokpitu lub odsunięcie siodełka."
                    ),
                )
            )

        # -------------------------------------------------------------
        # 3. COCKPIT & STEM (Zasięg, mostek, kąt łokcia i tułowia)
        # -------------------------------------------------------------
        elbow = cycle_stats.elbow_angle_median
        torso = cycle_stats.torso_angle_median
        cfg_elbow = self.config.elbow_angle

        if elbow > cfg_elbow.max_val:
            # Elbow completely straight (>165 deg) -> reach too long or cockpit too low
            recs.append(
                Recommendation(
                    category=RecommendationCategory.COCKPIT_REACH_STACK,
                    parameter_name="Kąt ugięcia łokcia",
                    measured_value=round(elbow, 1),
                    target_range=(cfg_elbow.min_val, cfg_elbow.max_val),
                    unit="deg",
                    action_summary="Skróć zasięg (reach): krótszy mostek o 10-20 mm lub podkładki pod mostek",
                    adjustment_mm=-10.0,
                    confidence=ConfidenceLevel.HIGH,
                    rationale=(
                        f"Kąt łokcia wynosi {elbow:.1f}° (ręce prawie całkowicie wyprostowane). "
                        "Brak naturalnego zgięcia uniemożliwia amortyzację wstrząsów, przenosi wibracje "
                        "bezpośrednio na barki i kark oraz wywołuje nadmierny nacisk na dłonie (drętwienie)."
                    ),
                    alternatives=(
                        "Krok 1: Wymień mostek na krótszy o 10–20 mm. "
                        "Krok 2: Dołóż podkładkę dystansową (spacer) pod mostek lub odwróć mostek na plus."
                    ),
                )
            )
        elif elbow < cfg_elbow.min_val:
            recs.append(
                Recommendation(
                    category=RecommendationCategory.COCKPIT_REACH_STACK,
                    parameter_name="Kąt ugięcia łokcia",
                    measured_value=round(elbow, 1),
                    target_range=(cfg_elbow.min_val, cfg_elbow.max_val),
                    unit="deg",
                    action_summary="Rozważ wydłużenie mostka o 10 mm (kokpit może być za ciasny)",
                    adjustment_mm=10.0,
                    confidence=ConfidenceLevel.LOW,
                    rationale=(
                        f"Kąt łokcia wynosi {elbow:.1f}° (dość mocne zgięcie przy tułowiu {torso:.1f}°). "
                        "Jeśli czujesz się ściśnięty w klatce piersiowej, kokpit może być nieco za krótki."
                    ),
                    alternatives="Jeśli pozycja jest dla Ciebie w 100% wygodna, zmiana nie jest konieczna.",
                )
            )

        # -------------------------------------------------------------
        # 4. FOOT ANKLE DYNAMICS (Praca stóp w DMP)
        # -------------------------------------------------------------
        foot_bdc = cycle_stats.foot_angle_bdc_median
        cfg_foot = self.config.foot_angle_bdc

        if foot_bdc > cfg_foot.max_val:
            recs.append(
                Recommendation(
                    category=RecommendationCategory.PEDALING_TECHNIQUE,
                    parameter_name="Kąt stopy w DMP (nachylenie pięta-palce)",
                    measured_value=round(foot_bdc, 1),
                    target_range=(cfg_foot.min_val, cfg_foot.max_val),
                    unit="deg",
                    action_summary="Zwróć uwagę na technikę: palce mocno opuszczone (tiptoeing)",
                    adjustment_mm=None,
                    confidence=ConfidenceLevel.MEDIUM,
                    rationale=(
                        f"Kąt stopy w DMP wynosi {foot_bdc:.1f}° (palce skierowane mocno w dół). "
                        "Tzw. 'pedałowanie z palców' bywa podświadomą próbą dosięgnięcia do zbyt wysokiego "
                        "siodełka lub wskazuje na bloki przesunięte za bardzo w stronę palców."
                    ),
                    alternatives=(
                        "Sprawdź pozycję bloków w butach – przesunięcie bloków w stronę śródstopia (do tyłu) "
                        "stabilizuje staw skokowy i odciąża łydki oraz ścięgno Achillesa."
                    ),
                )
            )
        elif foot_bdc < cfg_foot.min_val:
            recs.append(
                Recommendation(
                    category=RecommendationCategory.PEDALING_TECHNIQUE,
                    parameter_name="Kąt stopy w DMP",
                    measured_value=round(foot_bdc, 1),
                    target_range=(cfg_foot.min_val, cfg_foot.max_val),
                    unit="deg",
                    action_summary="Pięta opada poniżej osi pedału w DMP",
                    adjustment_mm=None,
                    confidence=ConfidenceLevel.MEDIUM,
                    rationale=(
                        f"W DMP pięta opada o {abs(foot_bdc):.1f}° poniżej poziomu. Zwykle świadczy to "
                        "o zbyt niskim siodełku lub nadmiernym rozciąganiu ścięgna Achillesa."
                    ),
                    alternatives="Podniesienie siodełka o kilka milimetrów zazwyczaj wyrównuje kąt stopy.",
                )
            )

        return FittingReport(
            recommendations=recs,
            cycle_stats=cycle_stats,
        )
