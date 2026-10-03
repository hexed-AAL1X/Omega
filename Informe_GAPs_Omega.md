# Informe técnico de GAPs — Proyecto Omega

**Curso:** Programación Concurrente y Distribuida (CC65)
**Proyecto revisado:** `Omega-main.zip` — predictor de éxito de proyectos de desarrollo (ODS 17) con Random Forest en Go
**Fecha de revisión:** 3 de octubre de 2026
**Tipo de revisión:** análisis estático manual del código + inspección de datos y resultados incluidos en el ZIP

---

## 1. Resumen ejecutivo

Omega es un proyecto **bien encaminado**: usa solo la biblioteca estándar de Go, tiene dos casos de estudio concurrentes (entrenamiento de un Random Forest y agregación de 1,25 M de registros), un diseño que garantiza resultados reproducibles (semilla por árbol), verificación formal con Spin y un benchmark con media recortada. La concurrencia en sí está bien planteada y **no encontré condiciones de carrera ni deadlocks** en la lectura del código.

Los GAPs más relevantes no están en el algoritmo concurrente sino alrededor de él:

| # | GAP principal | Severidad | Sección |
|---|---|---|---|
| 1 | **No existe componente distribuido.** Todo corre en una sola máquina, en un solo proceso. El curso es "Concurrente *y Distribuida*". | Alta | §9 |
| 2 | **Cero pruebas automatizadas** (no hay ningún `*_test.go`), ni CI. Toda la evidencia de correctitud es manual. | Alta | §6 |
| 3 | **Validez del modelo:** `duration_years` es probablemente una variable *ex post* (fuga de información) y el split es aleatorio, no por país/tiempo. | Alta | §7 |
| 4 | **Afirmaciones del informe PC2 no respaldadas por evidencia** o sobredimensionadas (carrera sin mutex, "bosque idéntico", "escalabilidad débil", capturas pendientes). | Media | §8, §10 |
| 5 | **Fragilidad de los workers:** sin `recover`, sin propagación de errores en el RF, sin cancelación (`context`). | Media | §3 |
| 6 | **Instrumentación de recursos sesgada** (`heap pico` incluye datos residentes; el CPU incluye el GC final). | Media | §5 |
| 7 | **Modelo Promela desalineado con el código Go** y `verificar.sh` que nunca falla (`\|\| true`). | Media | §8 |
| 8 | **Descarga de datasets sin verificación de integridad**, dependencias de Python sin declarar. | Media | §4 |

### Cuadro de calificación

| Dimensión | Nivel | Comentario breve |
|---|---|---|
| Diseño concurrente (Go) | 🟢 Bueno | Worker pool y fan-out/fan-in correctos; sin estado compartido mutable. |
| Robustez / manejo de errores | 🟠 Mejorable | Pánicos no capturados, validación de flags casi nula. |
| Calidad de código | 🟠 Mejorable | Mucha duplicación entre `cmd/*`, sin tests ni documentación de API. |
| Seguridad | 🟠 Mejorable | Superficie pequeña (herramienta local), pero sin verificación de descargas. |
| Metodología experimental | 🟠 Mejorable | Una sola máquina, eficiencia calculada contra hilos lógicos, métricas de memoria sesgadas. |
| Validez del modelo ML | 🔴 Débil | Posible fuga, validación no independiente, una sola partición. |
| Verificación formal | 🟡 Aceptable | Buenos modelos y contraejemplo; pero abstracción divergente y script no bloqueante. |
| Dimensión distribuida | 🔴 Ausente | Sin RPC, sin nodos, sin tolerancia a fallos. |
| Reproducibilidad / DevOps | 🟠 Mejorable | Sin `requirements.txt`, ETL solo en notebook, sin CI. |

---

## 2. Alcance y metodología

**Revisado:** 70 archivos del ZIP — 11 archivos Go (~1 100 líneas), 4 scripts Python, `EDA.ipynb` (ETL), 3 modelos Promela + resultados de Spin, `README.md`, `docs/informe_pc2.md`, CSV de resultados y los datasets incluidos (`model_table.csv/parquet`).

**Método:** lectura línea a línea del código, contraste del informe PC2 contra el código y los CSV de resultados, y cálculos propios sobre `model_table.csv` (porcentajes de nulos, distribución de la etiqueta, etc.).

**Limitaciones (importante):**

- En mi entorno **no había toolchain de Go**, así que **no ejecuté** `go build`, `go vet`, `go test -race` ni los benchmarks. Los hallazgos son de análisis estático; las afirmaciones sobre `-race` del informe no pude reproducirlas (ver §10).
- No dispongo de los datasets originales (PPD, Financing, WDI) ni del codebook de AidData; las observaciones sobre significado de variables deben validarse contra el codebook.
- Los números de rendimiento que cito son los que el equipo dejó en `resultados/`.

**Escala de severidad:** **Alta** = compromete la validez o la entrega; **Media** = riesgo real o deuda que conviene resolver; **Baja** = mejora de calidad / buena práctica.

---

## 3. Patrones de concurrencia (Go)

### 3.1 Qué está bien

- **Estado aleatorio no compartido:** cada árbol crea su propio `rand.Rand` con `seed + i` (`rfcore/forest.go:12-15`). Esto hace el resultado independiente del orden de ejecución — decisión excelente.
- **Datos compartidos solo en lectura:** `X` e `y` no se mutan durante el entrenamiento.
- **Fan-in del pipeline sin contención:** cada worker agrega en un mapa local y envía un parcial (`pipeline/financing.go:204-223`); la sincronización es mínima.
- **Cierre correcto del canal `parciales`:** una goroutine auxiliar hace `wg.Wait()` y `close()`; el canal tiene capacidad `workers`, por lo que ningún worker se bloquea al enviar.
- **Lectura de `primerE` después del cierre del canal:** el orden está garantizado por `happens-before` (`Unlock` → `wg.Done` → `wg.Wait` → `close` → `range` termina). Sin carrera.

### 3.2 Hallazgos

#### CONC-01 · Mutex y contador `done` redundantes en el worker pool — *Baja (pero afecta la narrativa académica)*
**Ubicación:** `rfcore/forest.go:59-82`

Cada goroutine escribe en `trees[i]` con índices **disjuntos** (cada `i` sale una sola vez del canal), por lo que el `Mutex` no protege nada necesario. El único dato que se muta de forma compartida es `done`, y además es redundante: `wg.Wait()` ya garantiza que todos los trabajos terminaron. El `panic` de la línea 81 es inalcanzable en la práctica.

**Impacto:** el informe (§i) concluye que "quitar el mutex introduce la condición de carrera". Eso es cierto *para el modelo Promela* (que incrementa `done`), pero **no para el código Go real**: sin mutex, `trees[i] = tree` no tendría carrera; solo `done++` la tendría. La conclusión está sobredimensionada.

**Recomendación:** (a) eliminar `mu` y `done`, y documentar en comentario por qué las escrituras por índice no requieren exclusión; o (b) conservar un contador de progreso con `atomic.Int64`. En ambos casos, ajustar el texto del informe.

#### CONC-02 · Sin `recover` ni propagación de errores en los workers — *Media*
**Ubicación:** `rfcore/forest.go:67-76`, `pipeline/financing.go:206-222`

Un `panic` dentro de `trainTree` (p. ej., índice fuera de rango por datos corruptos) termina todo el proceso sin posibilidad de manejo. `TrainConcurrent` no devuelve `error`. En el pipeline sí se registra `primerE`, pero los workers siguen consumiendo bloques tras el primer fallo y se devuelve un resultado parcial junto al error.

**Recomendación:** firma `TrainConcurrent(ctx, ...) (*Forest, error)`, `defer recover()` por worker que registre el primer error con `sync.Once` y cancele el `context`; los workers comprueban `ctx.Err()` antes de tomar el siguiente trabajo (ver Apéndice A.1).

#### CONC-03 · Sin cancelación ni timeouts (`context`) — *Baja/Media*
Ninguna operación larga (entrenar 200 árboles tarda ~22 s secuencial) puede interrumpirse limpiamente. Es el patrón estándar de Go para concurrencia y suele evaluarse en un curso de este tipo.

#### CONC-04 · Granularidad gruesa y techo teórico del speedup — *Baja*
**Ubicación:** `rfcore/forest.go:53-57`

La unidad de trabajo es un árbol completo. Con 50 árboles y 8 workers, el camino crítico es `⌈50/8⌉ = 7` árboles → speedup máximo teórico **7,14×** (no 8×); con 16 workers, `⌈50/16⌉ = 4` → 12,5×. Además, los árboles no cuestan lo mismo (profundidad y datos distintos), lo que añade desbalance de cola ("tail effect").

**Recomendación:** medir y reportar la dispersión del tiempo por árbol; para el análisis de escalabilidad usar más árboles (≥ 200) o paralelizar también la evaluación de candidatos a split.

#### CONC-05 · Algoritmo intensivo en asignaciones → el GC limita la escalabilidad — *Media*
**Ubicación:** `rfcore/tree.go:61-62, 85-94`

`bestSplit` ordena los índices **por cada feature candidata y por cada nodo** con `sort.Slice` (reflexión + closure), y cada nodo reserva `sorted`, `left` y `right` de tamaño `n`. El resultado: 161 MB asignados para 50 árboles y, en el pipeline, los CPU-segundos casi se duplican al pasar de 1 a 8 workers (5,97 → 12,83 s, `resultados/pipeline_workers.csv`). El informe lo atribuye a "contención de caché, más asignaciones y más GC", lo cual es plausible pero no está medido.

**Recomendación:** (1) preordenar cada feature una sola vez por árbol; (2) reutilizar buffers por worker (un `builder` por goroutine o `sync.Pool`); (3) partición in-place; (4) usar `slices.SortFunc`. Verificar el efecto con `GODEBUG=gctrace=1` y `go test -bench -benchmem`.

#### CONC-06 · Partición del CSV por `\n` ignora saltos de línea dentro de campos entrecomillados — *Media (riesgo latente)*
**Ubicación:** `pipeline/financing.go:160-180`

`partir` corta el buffer en el siguiente `\n`. Si algún campo de texto contiene un salto de línea entre comillas (válido en RFC 4180), un bloque quedaría partido y `encoding/csv` fallaría o, peor, desalinearía columnas. **En este dataset no se manifiesta** (0 filas inválidas y el resultado coincide con la versión secuencial, que lee el archivo completo con el parser correcto), pero el código no lo garantiza para otros archivos.

**Recomendación:** documentar la suposición; añadir una verificación explícita (`Filas` concurrente == `Filas` secuencial, ya existe parcialmente en `Iguales`); o cambiar a un productor que lea con `csv.Reader` en streaming y envíe lotes de registros por el canal.

#### CONC-07 · Lectura completa del archivo en memoria y sin solapamiento E/S–CPU — *Media*
**Ubicación:** `cmd/pipeline/main.go:41`

`os.ReadFile` carga 465 MB y recién entonces comienza el cómputo; no hay streaming ni solapamiento de lectura con parseo. Es una de las razones del heap de ~910 MB y limita el uso del patrón *pipeline* real (etapas conectadas por canales).

#### CONC-08 · Retención de memoria por subcadenas con `ReuseRecord` — *Baja*
**Ubicación:** `pipeline/financing.go:91, 106, 113-115`

Con `csv.Reader`, los campos de un registro son subcadenas de **una sola cadena por fila**. Usarlos como clave de mapa (`Clave.Receptor`, `Donantes[...]`) mantiene viva la fila completa (~370 B promedio). En este volumen el impacto es pequeño (decenas de miles de claves), pero el patrón escala mal.

**Recomendación:** `strings.Clone(rec[c.receptor])` al insertar claves (Go ≥ 1.20).

#### CONC-09 · Suma de flotantes no asociativa → resultados no idénticos bit a bit entre configuraciones — *Baja*
**Ubicación:** `pipeline/financing.go:130-150, 247-268`

Con distinto número de workers/bloques el orden de suma cambia, por eso `Iguales` usa tolerancia relativa 1e-6. Es razonable, pero el informe afirma "mismos conteos y sumas" sin matizar. Además `Iguales` compara solo el *tamaño* del conjunto de donantes, no su contenido.

**Recomendación:** fusionar parciales en orden determinista (por índice de bloque) o acumular en enteros (centavos) / Kahan; comparar conjuntos de donantes completos.

#### CONC-10 · La misma semilla alimenta el split, el submuestreo y el árbol 0 — *Baja*
**Ubicación:** `rfcore/dataset.go:154`, `rfcore/forest.go:13` (`seed+0`), `cmd/escalabilidad/main.go:137`

`rand.NewSource(seed)` se usa para la permutación del split, para `submuestra` y para el RNG del árbol 0: producen **la misma secuencia pseudoaleatoria**, introduciendo correlación innecesaria. Usar derivación explícita (`seed`, `seed+1_000_003`, …) o un `SplitMix64` por propósito.

#### CONC-11 · Predicción secuencial — *Baja*
`Forest.PredictAll` (`forest.go:94-100`) recorre las muestras en un solo hilo. Con 2 783 filas es irrelevante, pero es el candidato natural para un segundo patrón (paralelismo de datos por lotes) y para demostrar *speedup* en inferencia.

#### CONC-12 · Naming: lo que se llama "pipeline" es en realidad fork-join / map-reduce — *Baja (conceptual)*
`partir` se ejecuta completo antes de lanzar workers y el canal `tareas` se prellena y cierra: actúa como cola, no como etapa conectada. Es un patrón válido (fork-join con reducción), pero conviene nombrarlo correctamente en el informe o implementar un pipeline real (lector → parseadores → agregador) conectado por canales con *backpressure*.

---

## 4. Seguridad

La superficie de ataque es pequeña (herramientas de línea de comandos locales, sin red ni datos personales), por lo que no hay vulnerabilidades críticas. Sí hay prácticas que conviene corregir:

| ID | Hallazgo | Ubicación | Sev. |
|---|---|---|---|
| SEC-01 | **Descargas sin verificación de integridad.** Se descargan ZIP de AidData, GitHub y World Bank (por HTTPS, bien) sin comprobar hash SHA-256 ni tamaño esperado. | `download_datasets.py:53-72` | Media |
| SEC-02 | **ZIP parcial tratado como válido.** Se descarga directamente al nombre final; si se corta la descarga, en la siguiente ejecución `zip_path.is_file()` es verdadero y falla `BadZipFile`. | `download_datasets.py:114-119` | Media |
| SEC-03 | **Extracción:** usa `Path(info.filename).name` (mitiga *zip-slip*, buena práctica ✔), pero no limita el tamaño descomprimido (zip bomb) y dos entradas con el mismo nombre en carpetas distintas se sobrescriben. | `download_datasets.py:75-90` | Baja |
| SEC-04 | **Validación de entradas CLI casi inexistente.** `-trees 0` produce un bosque vacío que **predice siempre "éxito" sin error** (`majority(0,0)` devuelve 1); `-trees -1` provoca `panic` en `make`; `-workers 100000000` lanza esa cantidad de goroutines; `-trim ≥ 0.6` hace `panic` en `TrimmedMean`; `-reps 0` produce NaN. | `cmd/*/main.go`, `rfcore/metrics.go:62-68`, `rfcore/tree.go:122` | Media |
| SEC-05 | **Memoria sin límite:** `os.ReadFile` sobre cualquier ruta; un archivo equivocado o malicioso puede agotar RAM. | `cmd/pipeline/main.go:41` | Baja |
| SEC-06 | **Dependencias de Python sin declarar ni fijar** (`polars`, `matplotlib`, `seaborn`, `pandas`/`pyarrow` para `to_pandas`). No hay `requirements.txt`, `pyproject.toml` ni lockfile → riesgo de cadena de suministro y de irreproducibilidad. | raíz del repo | Media |
| SEC-07 | **Fugas menores de información / higiene:** ruta absoluta con nombre de usuario en la salida del notebook (`/home/aal1x/repo/Omega/...`); *User-Agent* falsificado como `Mozilla/5.0 Nexora-dataset-downloader` (nombre de otro proyecto). | `EDA.ipynb:901`, `download_datasets.py:55` | Baja |
| SEC-08 | **Licencias y gobierno de datos:** se versionan derivados de PPD/WDI (parquet de 14 MB, CSV) sin `LICENSE` ni sección de atribución/licencia de las fuentes. | raíz del repo | Baja/Media |
| SEC-09 | **Sin análisis automático de seguridad ni de dependencias** (`govulncheck`, `gosec`, `staticcheck`, `pip-audit`). | — | Baja |

**Puntos a favor:** Go solo con biblioteca estándar (cero dependencias externas), HTTPS en todas las fuentes, sin credenciales ni *secrets* en el repositorio (busqué patrones `key/token/password/secret`), datos agregados sin información personal.

---

## 5. Instrumentación y metodología experimental

### 5.1 Problemas de la medición de recursos (`recursos/recursos.go`)

| ID | Problema | Efecto | Sev. |
|---|---|---|---|
| EXP-01 | `cpu1` se lee **después del segundo `runtime.GC()`** (líneas 80-81). | El CPU del GC forzado se atribuye a la función medida; infla `CPUSegundos` y `UsoCPU`. | Media |
| EXP-02 | **"Heap pico" incluye el heap residente previo** (el archivo de 465 MB ya cargado, el dataset, etc.), no el incremento de la función. Evidencia: en `pipeline_tamano.csv`, con solo el 25 % de los datos el "heap pico" es **578 MB**, porque el archivo completo está en memoria. | La métrica no es comparable entre configuraciones; el aumento por concurrencia (+1,6 %) queda oculto. | Media |
| EXP-03 | Se muestrea `/memory/classes/heap/objects:bytes` cada 10 ms; esa métrica incluye **basura aún no liberada**, no solo objetos vivos. | No es el pico de memoria viva; picos cortos se pierden. Mejor `/gc/heap/live:bytes` + `HeapInuse` y restar la línea base. | Media |
| EXP-04 | `UsoCPU` divide por `runtime.NumCPU()` y no por `GOMAXPROCS` (línea 89). | Distorsiona el porcentaje si se restringe `GOMAXPROCS` o hay *cgroups*. | Baja |
| EXP-05 | La goroutine muestreadora compite con los workers (sobre todo con 1 worker). | Sesgo pequeño pero sistemático. | Baja |

### 5.2 Diseño experimental

- **EXP-06 · Eficiencia calculada contra hilos lógicos.** `eficiencia = speedup / workers`. Con 16 workers en un i5-1135G7 (4 núcleos físicos / 8 hilos) se reporta eficiencia 0,28 (RF) y 0,20 (pipeline), pero el techo físico es ~4–5×. Conviene reportar también la eficiencia respecto a **núcleos físicos** y marcar claramente la zona de sobresuscripción.
- **EXP-07 · Interpretación de Amdahl inconsistente.** El informe estima `p ≈ 0,95` usando N = 4 con un speedup medido con 8 workers. Aplicando la fórmula a *cada punto* del RF (`p = (1−1/S)/(1−1/N)`) se obtiene p = 0,95 (N=2), 0,92 (N=4), 0,88 (N=8) y 0,83 (N=16): **`p` decrece**, lo que indica que no es solo fracción serial, sino sobrecarga creciente (hyper-threading, memoria, GC). Un ajuste Amdahl/USL sobre todos los puntos sería más honesto.
- **EXP-08 · "Escalabilidad débil" mal usada.** Escalabilidad débil exige aumentar workers *proporcionalmente* al tamaño del problema; el experimento mantiene 8 workers fijos y varía los datos (eso es escalabilidad por tamaño). Además, el informe dice que el algoritmo "escala sin degradarse", pero en el RF el speedup **baja** de 4,08× a 3,67× al crecer los datos (`escalabilidad.csv`).
- **EXP-09 · Un solo entorno, no controlado.** Laptop con gobernador `powersave`, procesos de fondo y posible *thermal throttling*. El propio informe lo reconoce. Para el informe final: fijar gobernador `performance`, `taskset`/`GOMAXPROCS`, calentamiento previo, y reportar intervalos de confianza (p. ej., con `benchstat`).
- **EXP-10 · El único ancla de comparación es Go-propio.** No hay comparación contra una referencia externa (p. ej., `scikit-learn` con `n_jobs`) que valide tanto la exactitud del RF como el orden de magnitud del tiempo.
- **EXP-11 · La verificación de equivalencia es débil.** Compara solo las **predicciones sobre el test** y solo en la repetición 0 (`cmd/benchmark/main.go:68`, `cmd/escalabilidad/main.go:86-95`). Dos bosques distintos pueden predecir igual. Se debería comparar la **estructura de los árboles** (`reflect.DeepEqual(f1.Trees, f2.Trees)`). En `cmd/pipeline/main.go:72-78` el modo secuencial se compara contra sí mismo (tautología) y se recalcula `Secuencial(parte)` en cada configuración.
- **EXP-12 · `TrimmedMean` con pocas repeticiones.** Con `reps=5` y `trim=0.1`, `k = int(5·0.1) = 0`: **no recorta nada** y nadie se entera. Validar `k ≥ 1` o avisar.

---

## 6. Calidad de código e ingeniería

| ID | Hallazgo | Ubicación | Sev. |
|---|---|---|---|
| QA-01 | **No hay ninguna prueba automatizada.** Ni unitarias (`gini`, `majority`, `imputeMedian`, `partir`, `combinar`), ni de propiedad (concurrente == secuencial), ni *golden tests*. Verificado: no existe ningún `*_test.go`. | todo el repo | **Alta** |
| QA-02 | **Sin CI** (GitHub Actions): ni `go vet`, `staticcheck`, `go test -race`, ni verificación Spin. | — | Media |
| QA-03 | **Duplicación entre comandos:** `f4`, `guardarCSV`, `listaInt`/`enteros`, `listaFloat`/`decimales`, `salirSi` están copiados en `cmd/benchmark`, `cmd/escalabilidad` y `cmd/pipeline`. | `cmd/*/main.go` | Media |
| QA-04 | **`os.Exit` dentro de la lógica** (no solo en `main`) y funciones `main` de 90–110 líneas con closures anidadas: impide pruebas y omite `defer`. | `cmd/benchmark`, `cmd/escalabilidad`, `cmd/pipeline` | Media |
| QA-05 | **Errores de escritura ignorados:** `w.Write(...)` sin comprobar y `defer f.Close()` sin revisar el error sobre un archivo de escritura → posible CSV truncado sin aviso. | `cmd/pipeline/main.go:145-172` (y equivalentes) | Baja |
| QA-06 | **`panic` en código de biblioteca** en vez de devolver `error`. | `rfcore/forest.go:81` | Baja |
| QA-07 | **Sin comentarios de documentación (godoc)** en ningún paquete ni función exportada; sin `doc.go`. | `rfcore/`, `pipeline/`, `recursos/` | Baja |
| QA-08 | **Idioma mezclado** en la API (`TrainConcurrent`, `Predict` vs. `Medir`, `Agregado`, `Secuencial`). | varios | Baja |
| QA-09 | **Hiperparámetros y constantes fijas en el código** (profundidad 10, `MinSamplesSplit` 20, `MinSamplesLeaf` 5, test 0,2, muestreo 10 ms, 8 bloques por worker); no se pueden variar desde CLI. | `rfcore/tree.go:21`, `cmd/*` | Baja |
| QA-10 | **`LoadCSV` con parámetro booleano posicional** (`soloVentana`), búsquedas en mapa por fila y por columna dentro del bucle, todo el CSV en memoria antes de filtrar. | `rfcore/dataset.go:32-135` | Baja |
| QA-11 | **El modelo no se puede guardar ni cargar** (`Forest` sin serialización) y **no hay interfaz de inferencia** para proyectos nuevos; el codificador one-hot tampoco se persiste (categorías no vistas darían todo ceros sin aviso). El README habla de un "sistema de predicción" pero solo existe el flujo entrenar-evaluar. | `rfcore/*` | Media |
| QA-12 | **ETL solo en notebook:** la construcción de `model_table` vive en `EDA.ipynb`; `evidencia_limpieza.py` replica parte de la lógica con diferencias (p. ej., el formato de fecha `%-m/%-d/%Y` está en el notebook y falta en el script) → dos fuentes de verdad que pueden divergir. | `EDA.ipynb`, `evidencia_limpieza.py:42-49` | Media |
| QA-13 | **Fin de línea inconsistente:** `download_datasets.py` y `.gitignore` en CRLF, el resto en LF. Añadir `.gitattributes`. | raíz | Baja |
| QA-14 | **Celda de EDA que recorta a cero** (`clip(lower_bound=0)`, celda 11) contradice la política declarada en el README ("los decommitments no se recortaron a cero"). Si alguien reutiliza ese `DataFrame`, se pierden las cancelaciones. | `EDA.ipynb` | Baja |
| QA-15 | **Toolchain inconsistente:** `go.mod` declara `go 1.22`, el informe reporta Go 1.26.1; sin versión fijada en CI. | `go.mod`, `docs/informe_pc2.md` | Baja |
| QA-16 | **`TrimmedMean`/`Evaluate` sin validaciones:** `trim ≥ 0,6` produce `panic`; `Evaluate` no comprueba `len(yTrue) == len(yPred)` y con `n=0` devuelve NaN. | `rfcore/metrics.go` | Baja |

---

## 7. Datos y validez del modelo de ML

> Este bloque es relevante aunque el curso sea de concurrencia: si el modelo no es válido, el *speedup* se mide sobre un problema cuyo resultado no es defendible.

#### ML-01 · `duration_years` probablemente usa información *ex post* — **Alta**
El README declara que el predictor usa "atributos conocidos **al inicio** del proyecto". Sin embargo, `duration_years` se deriva de `project_duration` del PPD, que (salvo que el codebook indique lo contrario) mide la duración **real** del proyecto, solo conocida al terminar. Además, un proyecto retrasado suele recibir peor calificación. En los datos hay señal débil (éxito 76,1 % con duración < 4 años vs. 71,4 % con ≥ 8 años), por lo que el efecto puede ser moderado, pero **conceptualmente es una fuga**.
**Recomendación:** verificar en el codebook; hacer un *ablation* (entrenar sin `duration_years`) y reportar ambos resultados; si es ex post, eliminarla o sustituirla por la duración *planificada*.

#### ML-02 · Validación no independiente y una sola partición — **Alta**
`TrainTestSplit` es una permutación aleatoria 80/20 (`dataset.go:153-170`), sin estratificar. Los proyectos de un mismo país-año comparten *exactamente* las mismas features de contexto (WDI/Financing) y pueden caer en train y test → estimación optimista. Solo se evalúa una partición con 2 783 filas; no hay validación cruzada ni intervalos de confianza (el error estándar de la *balanced accuracy* con ese tamaño ronda ±0,01–0,02).
**Recomendación:** `GroupKFold` por país (o split temporal: entrenar hasta 2008, probar 2009+), validación cruzada k-fold, e IC por *bootstrap*.

#### ML-03 · Valores faltantes masivos tratados con imputación por mediana — Media
Sobre las 13 915 filas `fila_ok`: `commitment_usd` 62,8 % vacío, `n_donors`/`usd_ods17` 40,1 %, `literacy` 23,9 %, `poverty` 16,2 %. Solo `commitment_usd` y `n_donors` tienen *flag* de ausencia (`dataset.go:24`); el resto se imputa sin marca. La imputación está bien hecha en un aspecto clave: **la mediana se calcula solo con train** (✔). Pero la ausencia es informativa (depende del donante y del año).
**Recomendación:** *flags* de ausencia para todas las columnas con > 5 % de nulos; evaluar *train* solo con filas completas como análisis de sensibilidad.

#### ML-04 · Dominio de un solo donante y sin métricas por segmento — Media
El Banco Mundial aporta 5 774 de 13 915 filas (41,5 %), y es el único con `commitment_usd`. El modelo podría estar aprendiendo "es del WB". No hay métricas por donante, región o década.

#### ML-05 · Métricas y línea base pobres — Media
Se reporta *balanced accuracy* = 0,619 (vs. 0,5 del azar). No hay ROC-AUC, PR-AUC, calibración, matriz de confusión ni importancia de variables, ni modelo de referencia simple (regresión logística / árbol único).

#### ML-06 · Particularidades del RF propio que pueden limitar el rendimiento — Baja/Media
- **Hoja prematura:** si ninguna de las `k = √45 ≈ 6` features candidatas ofrece un split válido, el nodo se convierte en hoja (`tree.go:56-59`); implementaciones estándar siguen probando otras features. Con ~20 columnas *one-hot* de baja varianza, es probable que esto ocurra a menudo *(hipótesis: conviene medir cuántos nodos terminan así)*.
- **Empates → clase 1:** `majority` usa `2*ones >= n`; con 50/100/200 árboles pueden darse empates que sesgan hacia "éxito" (`tree.go:122-127`).
- **Vocabulario categórico construido con train+test** (`dataset.go:76-88`): no es fuga de etiqueta, pero sí de información del conjunto de prueba.

---

## 8. Verificación formal (Promela / Spin)

**Fortalezas:** modelos pequeños y legibles; propiedades de seguridad **y** vivacidad (LTL `<> terminado` con equidad débil `-f`); **control negativo** (`worker_pool_sin_mutex`) con contraejemplo guardado — práctica muy valiosa.

| ID | Hallazgo | Ubicación | Sev. |
|---|---|---|---|
| FM-01 | **`verificar.sh` nunca falla.** Cada ejecución de `pan` termina con `\|\| true` y solo hace `grep` de líneas de resumen; si una regresión introduce `errors: 1`, el script sigue devolviendo 0. | `promela/verificar.sh:23` | Media |
| FM-02 | **Abstracción divergente del código Go:** el modelo usa centinelas `FIN` y un mutex sobre `trees[j]`/`done`; el código usa `close(jobs)` + `for range` y **no necesita** ese mutex (CONC-01). El modelo verifica un diseño ligeramente distinto del implementado. Debe documentarse la correspondencia modelo ↔ código. | `worker_pool.pml`, `forest.go` | Media |
| FM-03 | **Caminos no modelados:** error/`panic` de un worker, cancelación, bloque con error en el pipeline (`primerE`), canal `parciales` con capacidad distinta de `W`. | `pipeline.pml` | Media |
| FM-04 | **Instancia mínima** (4 árboles / 3 workers; 4 bloques / 2 workers). Es habitual y válido, pero conviene declarar el argumento de "pequeña hipótesis de alcance" y probar al menos una segunda configuración (p. ej., más workers que trabajos). | `*.pml` | Baja |
| FM-05 | **Abstracción de la agregación:** el pipeline suma bytes (`filas`), operación asociativa y conmutativa; en Go se suman `float64` (no asociativo, CONC-09). Aceptable, pero debe decirse. | `pipeline.pml` | Baja |
| FM-06 | **Resultados de Spin con la versión 6.5.2 y sin registro de comandos de compilación en el informe** (flags `-DSAFETY`, `-DNOCLAIM`, `-w24`). | `resultados/*.txt` | Baja |

---

## 9. Dimensión distribuida (GAP principal frente al nombre del curso)

El proyecto demuestra **concurrencia** en memoria compartida (goroutines + canales + mutex) en **una sola máquina y un solo proceso**. No hay:

- comunicación entre procesos o nodos (RPC, gRPC, sockets, colas de mensajes);
- partición de datos/trabajo entre nodos con un coordinador;
- tolerancia a fallos (reintentos, *timeouts*, nodos caídos, mensajes duplicados o perdidos);
- serialización del modelo o de los parciales para transportarlos;
- modelado en Promela de pérdida/duplicación de mensajes o de fallo parcial.

**Por qué es fácil de cubrir con lo que ya existe:**

1. **Pipeline de Financing → modelo coordinador/trabajadores.** `Agregado`/`Resultado` ya son un *monoide* (la función `combinar` es asociativa y conmutativa salvo redondeo). Cada nodo procesa un rango de bytes y devuelve su parcial; el coordinador los fusiona. Con `net/rpc` + `encoding/gob` (ambos en la biblioteca estándar) bastan ~150 líneas.
2. **Random Forest distribuido.** Cada nodo entrena un subconjunto de árboles con `seed + i` (el determinismo ya está resuelto), serializa `[]*Node` con `gob` y el coordinador ensambla `Forest`. La predicción también se puede repartir.
3. **Fallos:** *timeout* + reasignación del bloque/árbol a otro nodo (el trabajo es idempotente gracias a la semilla por árbol).
4. **Modelo Promela extendido** con canal con pérdida (`chan` + `if :: skip :: send fi`) y reintento, verificando "todo trabajo termina exactamente una vez".

Aun si el alcance acordado para la PC2 fuera solo concurrencia, conviene declararlo explícitamente en el informe y planificar este componente para la entrega final.

---

## 10. Documentación y consistencia del informe PC2

Contraste entre lo que afirma `docs/informe_pc2.md` y lo observado:

| Afirmación del informe | Observación | Acción |
|---|---|---|
| "El bosque concurrente es **idéntico** al secuencial… lo comprueban comparando las predicciones." | Solo se comparan predicciones del test y solo en la repetición 0 (EXP-11). Es evidencia necesaria pero no suficiente. | Comparar estructura de árboles. |
| "El diseño con mutex está libre de condición de carrera y que quitar el mutex la introduce." | En el código Go el mutex es redundante para `trees[i]` (CONC-01). | Matizar o eliminar mutex. |
| "`go run -race` terminó con 0 advertencias en ambos algoritmos." | No hay salida guardada en `resultados/` ni prueba automatizada; no pude reproducirlo sin toolchain. | Guardar log y automatizarlo en CI. |
| "El algoritmo concurrente escala con el volumen sin degradarse." | En RF el speedup cae de 4,08× a 3,67× (EXP-08). | Reformular. |
| "Escalabilidad débil / por volumen." | No es escalabilidad débil (workers fijos). | Renombrar. |
| "Con 16 workers… eficiencia 0,28 / 0,20." | Eficiencia calculada sobre hilos lógicos con sobresuscripción (EXP-06). | Añadir eficiencia vs. núcleos físicos. |
| "Heap pico ~910 MB… la concurrencia apenas lo sube." | Métrica incluye línea base (EXP-02/03). | Medir delta. |
| `*(Insertar capturas de la ejecución…)*` (×2) y placeholder de `git log`/Contributors. | **Entregables pendientes** en el propio documento. | Completar antes de entregar. |
| "Pipeline con fan-out / fan-in" | Es fork-join / map-reduce (CONC-12). | Aclarar terminología. |

**Verificaciones que sí coinciden** (hecho propio sobre los datos): `model_table.csv` tiene 20 623 filas; `fila_ok` = 13 915; tasa de éxito en `fila_ok` = 75,4 % (10 497 / 13 915); 80 % de 13 915 = 11 132 filas de train y 2 783 de test, igual que el informe. El cálculo de Amdahl (0,81 con N=8) también es aritméticamente correcto, aunque su interpretación debe revisarse (EXP-07).

**Otros aspectos del README:** la sección se titula `# TRABAJO`, no describe cómo instalar las dependencias de Python ni el orden completo de ejecución (descarga → notebook → exportar CSV → Go → gráficos), y mezcla documentación de datos con la del sistema.

---

## 11. Fortalezas a conservar

1. Determinismo por semilla de árbol: concurrente ≡ secuencial por diseño.
2. Cero dependencias externas en Go.
3. Imputación por mediana calculada **solo con train**.
4. Baseline secuencial real, media recortada y medición de CPU/memoria — más de lo que suele entregarse.
5. Verificación formal con control negativo y contraejemplo.
6. Limpieza de datos documentada con cifras (embudo de filtrado) y control de calidad en el notebook.
7. Mitigación de *zip-slip* en la extracción.
8. Resultados reproducibles guardados en `resultados/` y gráficos generados por script.

---

## 12. Plan de acción priorizado

### Prioridad 1 — antes de la entrega (1–2 días)
| Acción | GAPs |
|---|---|
| Completar capturas, `git log --graph` y Contributors en el informe | §10 |
| Guardar la salida de `go vet ./...` y `go test -race ./...` / `go run -race` en `resultados/` | §10, QA-02 |
| Crear pruebas básicas: `gini`, `majority`, `partir` (sin pérdida ni duplicados), `combinar`, equivalencia concurrente == secuencial con `reflect.DeepEqual` | QA-01, EXP-11 |
| Validar flags (`trees ≥ 1`, `workers ≥ 1`, `0 ≤ trim < 0,5`, `reps ≥ 1`) | SEC-04, QA-16 |
| Corregir redacción: mutex, "idéntico", "escalabilidad débil", eficiencia, Amdahl | §10 |
| Añadir `requirements.txt` (versiones fijas) y arreglar `verificar.sh` para que falle con `errors ≠ 0` | SEC-06, FM-01 |
| Ablation sin `duration_years` y reporte de ambos resultados | ML-01 |

### Prioridad 2 — robustez y metodología (3–5 días)
| Acción | GAPs |
|---|---|
| `TrainConcurrent(ctx, …) (*Forest, error)` con `recover` y cancelación; quitar mutex/`done` o usar `atomic` | CONC-01/02/03 |
| Corregir `recursos.Medir` (CPU antes del GC final, heap en delta y `/gc/heap/live:bytes`) | EXP-01/02/03 |
| Split por país o temporal + validación cruzada + IC; métricas adicionales | ML-02/05 |
| Extraer helpers comunes a `internal/` (CSV, listas, `f4`) y eliminar `os.Exit` fuera de `main` | QA-03/04 |
| CI en GitHub Actions: `go vet`, `staticcheck`, `go test -race`, `govulncheck`, Spin | QA-02, SEC-09 |
| Verificar hash SHA-256 y descargar a `.part` + renombrar | SEC-01/02 |
| Pasar el ETL del notebook a un script (`etl.py`) como única fuente de verdad | QA-12 |

### Prioridad 3 — valor agregado (1–2 semanas)
| Acción | GAPs |
|---|---|
| **Componente distribuido** con `net/rpc` + `gob` (coordinador + N trabajadores, reintentos y timeouts) | §9 |
| Extender Promela con pérdida de mensajes y fallo de nodo | §9, FM-03 |
| Optimizar `bestSplit` (preorden, buffers reutilizables) y volver a medir | CONC-05 |
| Pipeline real por streaming (lector → parseadores → agregador) | CONC-06/07/12 |
| Persistencia del modelo (`gob`/JSON) + comando `predict` | QA-11 |
| Comparativa contra `scikit-learn` y ajuste Amdahl/USL con todos los puntos | EXP-07/10 |

---

## Apéndice A — Fragmentos de referencia

### A.1 Worker pool robusto (solo biblioteca estándar)

```go
func TrainConcurrent(ctx context.Context, X [][]float64, y []int,
	nTrees, workers int, params TreeParams, seed int64) (*Forest, error) {

	if nTrees < 1 {
		return nil, fmt.Errorf("nTrees debe ser >= 1, recibido %d", nTrees)
	}
	workers = max(1, min(workers, nTrees))

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	jobs := make(chan int, nTrees)
	for i := 0; i < nTrees; i++ {
		jobs <- i
	}
	close(jobs)

	trees := make([]*Node, nTrees) // índices disjuntos: no requiere mutex
	var (
		wg       sync.WaitGroup
		once     sync.Once
		firstErr error
	)
	fail := func(err error) { once.Do(func() { firstErr = err; cancel() }) }

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					fail(fmt.Errorf("panic en worker: %v", r))
				}
			}()
			for i := range jobs {
				if ctx.Err() != nil {
					return
				}
				trees[i] = trainTree(X, y, params, seed, i)
			}
		}()
	}
	wg.Wait()

	if firstErr != nil {
		return nil, firstErr
	}
	if err := ctx.Err(); err != nil && errors.Is(err, context.Canceled) {
		// cancelado por el llamador
		return nil, err
	}
	return &Forest{Trees: trees}, nil
}
```

### A.2 Prueba de equivalencia estructural

```go
func TestConcurrenteIgualSecuencial(t *testing.T) {
	ds := datasetSintetico(500, 12, 1) // helper de prueba
	p := DefaultTreeParams(12)
	seq := TrainSequential(ds.X, ds.Y, 20, p, 42)
	for _, w := range []int{1, 2, 4, 8} {
		con, err := TrainConcurrent(context.Background(), ds.X, ds.Y, 20, w, p, 42)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(seq.Trees, con.Trees) {
			t.Fatalf("workers=%d: los árboles difieren", w)
		}
	}
}
```

### A.3 Evitar retención de memoria por subcadenas

```go
k := Clave{Receptor: strings.Clone(rec[c.receptor]), Anio: anio}
...
don := strings.Clone(rec[c.donante])
a.Donantes[don] = struct{}{}
```

### A.4 `verificar.sh` que realmente falla

```bash
verificar() {
  local modelo="$1" nombre="$2" binario="$3" esperado_errores="${4:-0}"; shift 4
  local salida
  salida="$(cd "$TMP" && "./$binario" "$@" 2>&1 || true)"
  echo "$salida" > "$OUT/${modelo}_${nombre}.txt"
  local errores
  errores="$(grep -oE 'errors: [0-9]+' <<<"$salida" | head -1 | awk '{print $2}')"
  if [ "${errores:-X}" != "$esperado_errores" ]; then
    echo "FALLO: $modelo/$nombre -> errors=$errores (esperado $esperado_errores)" >&2
    exit 1
  fi
}
# modelos correctos: esperado 0; worker_pool_sin_mutex: esperado 1 (control negativo)
```

### A.5 Split temporal para validar sin fuga de contexto

```go
// Entrenar con proyectos anteriores a cutoff y evaluar con los posteriores.
func TemporalSplit(ds *Dataset, yearCol int, cutoff float64) Split {
	var s Split
	for i, row := range ds.X {
		if row[yearCol] < cutoff {
			s.TrainX, s.TrainY = append(s.TrainX, row), append(s.TrainY, ds.Y[i])
		} else {
			s.TestX, s.TestY = append(s.TestX, row), append(s.TestY, ds.Y[i])
		}
	}
	imputeMedian(s.TrainX, s.TestX)
	return s
}
```

---

## Apéndice B — Comandos recomendados para reproducir esta revisión

```bash
go vet ./...
go install honnef.co/go/tools/cmd/staticcheck@latest && staticcheck ./...
go install golang.org/x/vuln/cmd/govulncheck@latest && govulncheck ./...
go test -race -count=1 ./...
go run -race ./cmd/concurrente -trees 50 -workers 8
go run -race ./cmd/pipeline -reps 1 -workers 4
GODEBUG=gctrace=1 go run ./cmd/concurrente -trees 50 -workers 8 2>&1 | tail -20
./promela/verificar.sh
pip install pip-audit && pip-audit -r requirements.txt
```
