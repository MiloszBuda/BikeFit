# BikeFit - AI-Powered Bike Fitting Analysis

Aplikacja do komputerowej analizy pozycji kolarza (bike fitting) na podstawie nagrań wideo (np. trenażer domowy).

## Architektura i zrealizowane etapy

### Etap 1: Pipeline „wideo → punkty”
- **Model detekcji pozy**: RTMPose Halpe26 (`rtmlib`) – zawiera 26 punktów kluczowych:
  - Stawy kończyn dolnych: biodra, kolana, kostki.
  - Punkty stóp: pięta (`heel`), duży palec (`big_toe`), mały palec (`small_toe`) – kluczowe dla analizy techniki pedałowania.
  - Górna część ciała: barki, łokcie, nadgarstki/dłonie, kark (`neck`), ucho/głowa (`ear`, `head`).
- **Kalibracja suportu (środka korby)**:
  - Wskazanie interaktywne na pierwszej klatce (kliknięcie myszką w oknie OpenCV + potwierdzenie Spacją/Enterem).
  - Lub bezpośrednie podanie współrzędnych przez CLI (`--crank X,Y`).
- **Warstwa abstrakcji (`BasePoseEstimator`)**:
  - Odseparowanie modelu od pipeline'u i kalkulacji kątowych.
- **Wizualizacja kontrolna**:
  - Generowanie pliku wideo MP4 z nałożonym szkieletem Halpe26, kolorystycznym rozróżnieniem lewej i prawej strony, celownikiem osi korby oraz licznikiem klatek i czasu.

### Etap 2: Filtracja, kalibracja skali i kąty
- **Filtracja sygnału**:
  - Zero-phase Butterworth lowpass filter 2. rzędu (`scipy.signal.filtfilt`) z odcięciem na 5 Hz (eliminuje jitter detekcji, zachowując nienaruszone szczyty DMP/GMP).
  - Maskowanie klatek o niskiej pewności (`confidence < threshold`) i interpolacja liniowa ze skrajnym wypełnieniem brzegów.
- **Kalkulacja kątów biomechanicznych**:
  - **Kąt kolana** (wewnętrzny $0^\circ-180^\circ$ oraz flexion $180^\circ - \text{kąt}$)
  - **Kąt biodra** (bark – biodro – kolano)
  - **Kąt tułowia** (nachylenie linii biodro–bark względem poziomu)
  - **Kąt łokcia** (bark – łokieć – nadgarstek)
  - **Kąt ramię–tułów** (łokieć – bark – biodro)
  - **Kąt kostki** (kolano – kostka – duży palec)
  - **Kąt stopy** (nachylenie pięta–palce względem poziomu)
- **Kalibracja skali (px → mm)**:
  - Obliczenie współczynnika skali w oparciu o fizyczną długość ramienia korby (`--crank-length`, domyślnie 172.5 mm).
- **Wykresy i eksport**:
  - Eksport serii czasowych kątów do CSV i Parquet (`*_angles.csv`, `*_angles.parquet`).
  - Generowanie 3-panelowego wykresu (`*_kinematics.png`) z zaznaczonym zakresem docelowym DMP dla kolana ($145^\circ-155^\circ$ / $25^\circ-35^\circ$ flexion).

## Szybki start

```bash
# Aktywacja wirtualnego środowiska
.venv\Scripts\activate

# 1. Pełna analiza z interaktywnym wskazaniem suportu na 1. klatce:
python -m bikefit.cli video.mp4 --output-dir ./wyniki --mode balanced

# 2. Analiza z bezpośrednio podaną osią korby i ramieniem 170 mm:
python -m bikefit.cli video.mp4 --output-dir ./wyniki --crank 640,720 --crank-length 170 --mode balanced

# 3. Wymuszenie analizy konkretnej strony kolarza:
python -m bikefit.cli video.mp4 --output-dir ./wyniki --side right --crank 640,720

# Dostępne tryby RTMPose:
#   --mode lightweight  (najszybszy)
#   --mode balanced     (optymalny, domyślny)
#   --mode performance  (maksymalna precyzja)
```

## Uruchamianie testów

```bash
pytest -v
```
