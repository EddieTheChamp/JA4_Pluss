# JA4+ Classification Prototype

Repoet er i praksis delt i to lag akkurat nå:

- `current/` inneholder den aktive prototypen
- toppnivåmapper som `Custom Database/`, `Create Dictionary/` og `Dictionary/` inneholder eldre data og hjelpefiler som fortsatt brukes som fallback etter flyttingen

Målet er fortsatt å samle aktiv kode i `current/` og aktiv data i `data/`, men i denne kopien er dataflyttingen ikke helt fullfort ennå. Koden er derfor satt opp til a fungere med bade ny og gammel plassering.

## Status i denne repoen

Aktiv prototypekode ligger i:

- [current/app](current/app)
- [current/core](current/core)
- [current/services](current/services)
- [current/io](current/io)
- [current/utils](current/utils)
- [current/tests](current/tests)

Kompatibilitets-wrappere ligger i:

- [app](app)

Datasett og referansefiler som faktisk finnes i denne kopien ligger forelopig i:

- [Custom Database](Custom%20Database)
- [Create Dictionary](Create%20Dictionary)
- [Dictionary](Dictionary)

## Onsket malstruktur

Dette er fortsatt anbefalt sluttstruktur:

```text
current/
  app/
  core/
  services/
  io/
  utils/
  tests/
data/
  datasets/
  local_db/
  models/
  references/
  samples/
legacy/
  README.md
```

Per na finnes `current/`, men `data/` og `legacy/` er ikke ferdig innfort i denne arbeidskopien.

## Arkitektur

Den aktive prototypen bestar av:

- `current/core/models.py` for standardiserte dataklasser
- `current/core/classifier.py` for den sentrale pipelinen
- `current/core/decision_engine.py` for eksplisitte beslutningsregler
- `current/services/local_matcher.py` for lokal exact match
- `current/services/random_forest.py` og `current/services/inference.py` for Random Forest
- `current/services/foxio_adapter.py` for FoxIO-stotte
- `current/services/loaders.py` for lasting av dataset, DB og fallback-stier
- `current/io/*` for innlesing og formattering

## Kjoring

Anbefalt entrypoint i denne repoen:

```powershell
& "C:\Users\Vegard\AppData\Local\Programs\Python\Python312\python.exe" -m current.app.main classify --ja4 "..." --ja4s "..."
```

Bakoverkompatibel entrypoint fungerer fortsatt:

```powershell
& "C:\Users\Vegard\AppData\Local\Programs\Python\Python312\python.exe" -m app.main classify --ja4 "..." --ja4s "..."
```

Hvis du bruker et virtuelt miljo, kan du erstatte denne banen med din egen Python-binær.
Merk: `py -3` peker pa denne maskinen til en Windows Store-alias som ikke lot seg bruke til testingen.

Batch fra fil nar `data/`-strukturen er pa plass:

```powershell
& "C:\Users\Vegard\AppData\Local\Programs\Python\Python312\python.exe" -m current.app.main classify-file --input "data\samples\single_test_input.json" --output "data\samples\single_test_result.json" --output-format json
```

Batch med dagens mapper peker vanligvis til en fil du legger selv inn:

```powershell
& "C:\Users\Vegard\AppData\Local\Programs\Python\Python312\python.exe" -m current.app.main classify-file --input "input.json" --output-format terminal
```

## Tester

Anbefalt testkommando:

```powershell
& "C:\Users\Vegard\AppData\Local\Programs\Python\Python312\python.exe" -m unittest discover -s current/tests -v
```

Merk: gammel testkommando via `tests/`-wrapper finnes ikke i denne kopien akkurat na.

## RF og lokal DB

Tren RF-modellen fra datasettet som ligger i repoet i dag:

```powershell
& "C:\Users\Vegard\AppData\Local\Programs\Python\Python312\python.exe" -m current.app.main train-rf --dataset "Custom Database\correlated_ja4_db_large.json" --model-output "data\models\ja4_random_forest.pkl"
```

Bygg lokal DB fra samme dataset:

```powershell
& "C:\Users\Vegard\AppData\Local\Programs\Python\Python312\python.exe" -m current.app.main build-local-db --dataset "Custom Database\correlated_ja4_db_large.json" --output "data\local_db\egenlagd_correlated_db.json"
```

Koden prover automatisk disse plasseringene i prioritert rekkefolge:

- ny struktur under `data/`
- eksisterende filer under `Custom Database/`
- eldre filer under `Create Dictionary/` og `Dictionary/`

Det betyr at prototypen kan kjores under overforingen, sa lenge minst ett gyldig dataset eller en lokal DB fortsatt finnes.

## Ekstra avhengigheter

For FoxIO/TShark-relatert arbeid:

- Tshark ma vare installert via Wireshark
- `C:\Program Files\Wireshark\` bor ligge i systemets `Path`

For Random Forest-delen trengs i tillegg Python-pakkene:

- `numpy`
- `pandas`
- `scikit-learn`

Uten disse pakkene kan prototypen fortsatt starte og bruke lokal DB/FoxIO-flyt, men RF-trening og RF-prediksjon blir markert som utilgjengelig.

## Om overforingen

Denne README-en er oppdatert for arbeidskopien i:

`C:\Users\Vegard\JA4+\JA4_Pluss`

Neste naturlige steg er a flytte aktive datafiler fra `Custom Database/` inn i `data/`, og eventuelt samle eldre eksperimenter i `legacy/`. Koden er allerede forberedt pa den overgangen via fallback-logikken i `current/services/loaders.py`.


python Modeller/pipeline.py --ja4 "t13d1516h2_8a2d1d4d_8a2d1d4d" --ja4s "t130200_1301_8a2d1d4d" --ja4t "2016_02_2016" --ja4ts "2016_03_2016"