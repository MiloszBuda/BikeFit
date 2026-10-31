# BikeFit - AI-Powered Bike Fitting Analysis

Kompleksowa aplikacja do komputerowej analizy pozycji kolarza (bike fitting) na trenażerze na podstawie nagrań wideo (RTMPose Halpe26 + Kinematyka + Silnik reguł biomechanicznych).

---

## Przegląd możliwości

1. **Estymacja pozy (RTMPose Halpe26)**:
   - Śledzenie 26 punktów kluczowych kolarza (stawy biodrowe, kolanowe, skokowe, pięty, palce stóp, barki, łokcie, nadgarstki, kark, głowa).
   - Lekki silnik inferencyjny ONNX Runtime (`rtmlib`) – bez ciężkich zależności PyTorch/MMCV.
   - Trzy tryby: `lightweight` (najszybszy), `balanced` (domyślny), `performance` (najwyższa dokładność).
2. **Kalibracja geometrii roweru**:
   - Wskazanie środka korby / suportu: interaktywne (kliknięcie na pierwszej klatce) lub z linii poleceń (`--crank X,Y`).
   - Automatyczna skala w milimetrach (`px/mm`) na podstawie zadanej długości ramienia korby (`--crank-length`, domyślnie 172.5 mm).
3. **Filtracja sygnału i kalkulacja kątów**:
   - Zerofazowy filtr dolnoprzepustowy Butterwortha 2. rzędu (5 Hz) usuwający jitter bez przesuwania fazy i bez spłaszczania DMP/GMP.
   - Interpolacja klatek o niskiej pewności detekcji.
   - Ciągłe serie czasowe kątów stawów: kolano (wewnętrzny i flexion), biodro, tułów, łokieć, ramię-tułów, kostka, nachylenie stopy.
4. **Segmentacja cykli pedałowania i statystyki**:
   - Automatyczne wykrywanie każdego pełnego obrotu korby.
   - Detekcja **Dolnego Martwego Punktu (DMP / BDC)** i **Górnego Martwego Punktu (GMP / TDC)** za pomocą analizy minimów/maksimów trajektorii.
   - Obliczenie kadencji (RPM), mediany kątów oraz odchylenia standardowego (miara powtarzalności ruchu).
   - Automatyczne odrzucanie cykli rozbiegowych i wygaszania (warmup / cooldown).
5. **Silnik reguł i doradca fittingowy**:
   - **Wysokość siodełka**: wyliczana z kąta zgięcia kolana w DMP ($25^\circ-35^\circ$) z dokładnym przelicznikiem na milimetry ($1^\circ \approx 5.5$ mm) i bezpiecznym limitem kroku (maks. 5-6 mm na raz).
   - **Długość korby**: analiza kąta zamknięcia biodra w GMP ($< 45^\circ$ sugeruje skrócenie ramion korby).
   - **Kokpit i mostek**: wnioskowanie z kąta ugięcia łokcia ($150^\circ-165^\circ$) i nachylenia tułowia.
   - **Technika stóp**: analiza opadania/zadzierania palców w DMP (pozycja bloków, tiptoeing).
   - Poziomy pewności zaleceń (Wysoka / Średnia / Niska) wraz z uzasadnieniem fizjoterapeutycznym i alternatywami.
   - Możliwość modyfikacji zakresów docelowych w plikach YAML.
6. **Wizualizacje i raporty**:
   - Wideo wyjściowe MP4 z nałożonym szkieletem Halpe26 i celownikiem suportu.
   - Wykresy przebiegu kątów w czasie z zaznaczonymi punktami DMP/GMP i strefami docelowymi (`*_kinematics.png`).
   - Kompletny raport w terminalu oraz plikach: Markdown (`*_report.md`), Text (`*_report.txt`) i JSON (`*_report.json`).

---

## Szybki start

```bash
# 1. Aktywacja środowiska wirtualnego:
.venv\Scripts\activate

# 2. Uruchomienie analizy z interaktywnym wskazaniem osi korby:
python -m bikefit.cli video.mp4 --output-dir ./wyniki --mode balanced

# 3. Uruchomienie bez okna GUI (podane współrzędne suportu i ramienia 170 mm):
python -m bikefit.cli video.mp4 --output-dir ./wyniki --crank 640,720 --crank-length 170 --mode balanced

# 4. Uruchomienie z własnym plikiem konfiguracyjnym zakresów YAML:
python -m bikefit.cli video.mp4 --output-dir ./wyniki --config my_ranges.yaml
```

---

## Wyniki analizy w katalogu wyjściowym

| Plik | Opis |
| :--- | :--- |
| `*_annotated.mp4` | Wideo z nałożonym szkieletem stawów i osią korby |
| `*_kinematics.png` | 3-panelowy wykres kątów w czasie z oznaczonymi DMP/GMP i strefą optymalną |
| `*_report.md` | Raport fittingowy w formacie Markdown z tabelami i zaleceniami |
| `*_report.txt` | Tekstowe podsumowanie diagnostyczne |
| `*_report.json` | Strukturyzowane dane JSON dla aplikacji frontendowych |
| `*_angles.csv` / `.parquet` | Serie czasowe kątów dla każdej klatki |
| `*_keypoints.csv` / `.parquet` | Surowe współrzędne 26 punktów kluczowych |

---

## Uruchamianie testów

```bash
pytest -v
```
Wszystkie 41 testów jednostkowych i integracyjnych wykonuje się lokalnie w ~3-4 sekundy.
