# BikeFit - AI-Powered Bike Fitting Analysis

[![Static Badge](https://img.shields.io/badge/DEMO-red?style=for-the-badge)](https://bikefit-milosz.streamlit.app/)

Kompleksowa aplikacja do komputerowej analizy pozycji kolarza (bike fitting) na trenażerze na podstawie nagrań wideo (RTMPose Halpe26 + Kinematyka + Silnik reguł biomechanicznych + Interfejs Webowy Streamlit).

---

## Przegląd możliwości

1. **Aplikacja Webowa (Streamlit)**:
   - Wgrywanie plików wideo (MP4, MOV, AVI) metodą drag & drop.
   - Interaktywny podgląd pierwszej klatki z suwakami do kalibracji osi korby (suportu) w czasie rzeczywistym.
   - Pasek postępu analizy.
   - 4-zakładkowy pulpit wyników:
     - **Porady i Rekomendacje**: karty z zaleceniami zmian w milimetrach, uzasadnieniem i poziomem pewności.
     - **Wideo ze szkieletem**: odtwarzacz wideo z naniesionym szkieletem 26 stawów (kodowanie H.264 z yuv420p zapewniające pełną zgodność z przeglądarkami).
     - **Wykresy kinematyki**: wykresy z oznaczonymi punktami DMP (dolny martwy punkt) i GMP (górny martwy punkt).
     - **Pobieranie**: natychmiastowe pobieranie raportu HTML, Markdown, JSON oraz danych CSV/Parquet.
2. **Estymacja pozy (RTMPose Halpe26)**:
   - Śledzenie 26 punktów kluczowych kolarza (stawy biodrowe, kolanowe, skokowe, pięty, palce stóp, barki, łokcie, nadgarstki, kark, głowa).
   - Lekki silnik inferencyjny ONNX Runtime (`rtmlib`) – bez ciężkich zależności PyTorch/MMCV.
   - Trzy tryby: `lightweight` (najszybszy), `balanced` (domyślny), `performance` (najwyższa dokładność).
3. **Kalibracja geometrii roweru**:
   - Wskazanie środka korby / suportu: interaktywne (w UI lub w oknie OpenCV) lub z CLI (`--crank X,Y`).
   - Automatyczna skala w milimetrach (`px/mm`) na podstawie zadanej długości ramienia korby (`--crank-length`, domyślnie 172.5 mm).
4. **Filtracja sygnału i kalkulacja kątów**:
   - Zerofazowy filtr dolnoprzepustowy Butterwortha 2. rzędu (5 Hz) usuwający jitter bez przesuwania fazy i bez spłaszczania DMP/GMP.
   - Interpolacja klatek o niskiej pewności detekcji.
   - Ciągłe serie czasowe kątów stawów: kolano (wewnętrzny i flexion), biodro, tułów, łokieć, ramię-tułów, kostka, nachylenie stopy.
5. **Segmentacja cykli pedałowania i statystyki**:
   - Automatyczne wykrywanie każdego pełnego obrotu korby.
   - Detekcja **Dolnego Martwego Punktu (DMP / BDC)** i **Górnego Martwego Punktu (GMP / TDC)**.
   - Obliczenie kadencji (RPM), mediany kątów oraz odchylenia standardowego (miara powtarzalności ruchu).
   - Odrzucanie cykli rozbiegowych i wygaszania (warmup & cooldown).
6. **Silnik reguł i doradca fittingowy**:
   - **Wysokość siodełka**: wyliczana ze zgięcia kolana w DMP ($25^\circ-35^\circ$) z przelicznikiem na milimetry ($1^\circ \approx 5.5$ mm) i bezpiecznym krokiem (maks. 5–6 mm).
   - **Długość korby**: analiza kąta zamknięcia biodra w GMP ($< 45^\circ$ sugeruje skrócenie korby).
   - **Kokpit i mostek**: wnioskowanie z kąta ugięcia łokcia ($150^\circ-165^\circ$) i nachylenia tułowia.
   - **Technika stóp**: analiza opadania/zadzierania palców w DMP (pozycja bloków).
   - Zastrzeżenie medyczne i poziomy pewności zaleceń.
7. **Wieloformatowe raporty**:
   - Samodzielny raport HTML z wbudowanymi wykresami w formacie base64 (gotowy do wysłania e-mailem).
   - Raport Markdown (`*_report.md`), tekstowy (`*_report.txt`) i ustrukturyzowany JSON (`*_report.json`).

---

## Jak uruchomić

### 1. Interfejs Webowy (Streamlit)

Najwygodniejszy sposób korzystania z aplikacji:

```bash
# Aktywacja środowiska wirtualnego:
.venv\Scripts\activate

# Uruchomienie aplikacji w przeglądarce:
streamlit run bikefit/ui/app.py
```
Aplikacja automatycznie otworzy się w przeglądarce pod adresem `http://localhost:8501`.

---

### 2. Tryb konsolowy (CLI / skrypt)

```bash
# Z interaktywnym wskazaniem osi korby:
python -m bikefit.cli video.mp4 --output-dir ./wyniki --mode balanced

# Z bezpośrednio podanymi współrzędnymi suportu i długością ramienia 170 mm:
python -m bikefit.cli video.mp4 --output-dir ./wyniki --crank 640,720 --crank-length 170
```

---

## Pliki generowane w katalogu wyjściowym

| Plik | Opis |
| :--- | :--- |
| `*_report.html` | Samodzielny, interaktywny raport HTML z wbudowanymi wykresami |
| `*_annotated.mp4` | Wideo ze szkieletem Halpe26 i celownikiem osi korby (zgodne z przeglądarkami H.264) |
| `*_kinematics.png` | 3-panelowy wykres kątów w czasie z oznaczonymi punktami DMP i GMP |
| `*_report.md` | Raport fittingowy w formacie Markdown |
| `*_report.json` | Ustrukturyzowane dane JSON dla aplikacji i baz danych |
| `*_angles.csv` / `.parquet` | Serie czasowe kątów dla każdej klatki |
| `*_keypoints.csv` / `.parquet` | Surowe współrzędne 26 punktów kluczowych |

---

## Uruchamianie testów

```bash
pytest -v
```
Wszystkie 44 testy wykonują się w ~4-5 sekund.
