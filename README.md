# BikeFit - AI-Powered Bike Fitting Analysis

Aplikacja do komputerowej analizy pozycji kolarza (bike fitting) na podstawie nagrań wideo (np. trenażer domowy).

## Architektura i Etap 1 (Wideo → Punkty kluczowe)

- **Model detekcji pozy**: RTMPose Halpe26 (`rtmlib`) – zawiera 26 punktów kluczowych:
  - Stawy kończyn dolnych: biodra, kolana, kostki
  - Punkty stopy: pięta (`heel`), duży palec (`big_toe`), mały palec (`small_toe`) – kluczowe dla kąta stopy i pedałowania
  - Górna część ciała: barki, łokcie, nadgarstki/dłonie, kark (`neck`), ucho/głowa (`ear`, `head`)
- **Kalibracja suportu (środka korby)**:
  - Wskazanie interaktywne na pierwszej klatce (kliknięcie myszką w oknie OpenCV + potwierdzenie Spacją/Enterem).
  - Lub podanie współrzędnych przez CLI (`--crank X,Y`).
- **Warstwa abstrakcji (`BasePoseEstimator`)**:
  - Pozwala na bezproblemową wymianę modelu na inny bez ruszania pipeline'u i reguł.
- **Eksport danych**:
  - CSV oraz Parquet (klatka, sygnatura czasowa, nazwa stawu, x, y, confidence score, współrzędne osi korby).
- **Wizualizacja kontrolna**:
  - Generowanie pliku wideo MP4 z nałożonym szkieletem Halpe26, celownikiem osi korby oraz HUD z numerem klatki i czasem.

## Szybki start

```bash
# Aktywacja wirtualnego środowiska
.venv\Scripts\activate

# Przetwarzanie wideo z interaktywnym wskazaniem środka korby:
python -m bikefit.cli video.mp4 --output-dir ./results

# Przetwarzanie z podanym środkiem korby (bez otwierania okna):
python -m bikefit.cli video.mp4 --output-dir ./results --crank 640,720 --mode balanced

# Tryby RTMPose:
#   --mode lightweight  (najszybszy)
#   --mode balanced     (optymalny)
#   --mode performance  (maksymalna precyzja)
```

## Uruchamianie testów

```bash
pytest -v
```
