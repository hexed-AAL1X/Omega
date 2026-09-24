# TRABAJO

## Integrantes

- Leonardo Leoncio Bravo Ricapa
- Raul Andres Cerreño Zevalios
- Sergio Andres Saavedra Cervera

### Descripción del dataset 

Para poder llevar a buen término el sistema de predicción del éxito de proyectos de desarrollo sostenible, fundamentado en el ODS 17 (Alianzas para lograr los objetivos), 
fue preciso integrar tres fuentes públicas de cooperación internacional y de indicadores ODS, obteniéndose finalmente un conjunto de datasets estructurados que representan distintas 
dimensiones del mismo fenómeno: la evaluación ex post del proyecto, el volumen de financiación alineada a los Objetivos de Desarrollo Sostenible y el contexto socioeconómico del país receptor en el año de inicio.


Los principales datasets recopilados incluyen:

**Etiquetas y atributos de proyectos:**
- PPD2_Jan_21_2022.csv
  
**Volumen de cooperación y ODS 17:**
- FinancingtotheSDGsDataset_v1.0.csv
  
**Contexto socioeconómico del país:**

- SDGData.csv
- SDGCountry.csv


El volumen y la estructura de los datos permiten aplicar concurrencia en:
- Procesamiento simultáneo de lotes de proyectos evaluados
- Agregación paralela de 1 252 036 filas de Financing
- Cruce concurrente por ISO3 y año de inicio
- Análisis distribuido de cobertura histórica y de nulos

  
Esto permite reducir tiempos de procesamiento y mejorar la escalabilidad del sistema.


Los datasets finales fueron los siguientes:

**model_table.parquet**

| **Columna** | **Tipo de dato** | **Descripción** |
|---|---|---|
| `project_id` | string | ID único del proyecto evaluado. |
| `donor` | string | Agencia evaluadora (WB, KfW, DFID, etc.). |
| `iso3` | string | Código de país ISO3. |
| `start_year` | int | Año de inicio del proyecto (startyear de AidData). |
| `sector_code` | string | Código CRS del sector. |
| `sector_description` | string | Descripción del sector. |
| `duration_years` | float | Duración del proyecto en años. |
| `commitment_usd` | float | Compromiso neto en USD (principalmente Banco Mundial). |
| `wb_region` | string | Región del Banco Mundial. |
| `wb_income` | string | Grupo de ingreso del país. |
| `gdp_pc` | float | PIB per cápita del país al inicio del proyecto. |
| `elec_access` | float | Acceso a electricidad (%). |
| `child_mort` | float | Mortalidad infantil. |
| `n_donors` | int | Número de donantes distintos en el país-año. |
| `usd_ods17` | float | Monto atribuido al ODS 17. |
| `rating` | float | Nota global del proyecto en escala de 1 a 6. |
| `success` | int | Variable objetivo: 1 si `rating ≥ 4`; 0 en caso contrario. |
| `fila_ok` | bool | Indica si el país es válido y el año se encuentra entre 1990 y 2017. |


**Columnas de contexto construidas (Banco Mundial y Financing)**

| **Columna** | **Tipo de dato** | **Descripción** |
|---|---|---|
| `gdp_pc_ppp` | float | PIB per cápita en términos de paridad de poder adquisitivo (PPA). |
| `gdp_pc_growth` | float | Crecimiento del PIB per cápita (%). |
| `unemp` | float | Tasa de desempleo (%). |
| `urban_pct` | float | Población urbana (%). |
| `internet_pct` | float | Uso de internet (%). |
| `poverty` | float | Población en situación de pobreza según el umbral de $1.90/día (%). |
| `literacy` | float | Tasa de alfabetización adulta (%). |
| `n_aid_projects` | int | Número de actividades de ayuda registradas en el país-año. |
| `usd_total_net` | float | Compromiso neto total de cooperación. |
| `usd_total_gross` | float | Compromiso bruto total, considerando únicamente montos mayores a 0. |
| `n_projects_ods17` | int | Número de actividades con `goal_17 > 0`. |
| `n_decommitments` | int | Número de cancelaciones o descompromisos (compromiso < 0). |
| `in_financing_window` | bool | Indica si el inicio del proyecto se encuentra dentro del periodo de financiación 2000–2013. |
| `has_wb_context` | bool | Indica si el proyecto cuenta con información contextual del Banco Mundial. |
| `has_financing` | bool | Indica si el proyecto cuenta con información de cooperación/financiamiento. |


El caso de uso priorizado se corresponde con un predictor del éxito de proyectos de desarrollo sostenible cuya fuente de etiquetas es el PPD de AidData
y cuyo contexto proviene de Financing to the SDGs y de los indicadores ODS del Banco Mundial. El objetivo del sistema reside en comparar atributos conocidos al inicio del
proyecto (donante, sector, país, año y entorno del receptor) con el resultado ex post, finalizando en una clasificación de éxito cuando la nota global es mayor o igual a 4.


Para este trabajo se da forma a un dataset principal de modelado y a dos paneles de apoyo. El primero es model_table.parquet (también model_df en el notebook), que almacena una 
fila por proyecto evaluado, con la etiqueta success, el donante, el ISO3, el año de inicio, el sector, la duración en años, el compromiso y el contexto del país. Los paneles de apoyo son 
el del Banco Mundial (país x año, indicadores ODS) y el de Financing (país x año, n_donors, usd_total_net, usd_ods17).


La construcción de este dataset fue realizada por medio del notebook EDA.ipynb, que posee como función el procesamiento de los archivos CSV originales, 
la normalización de países a ISO3, la conversión de la duración de días a años y la generación de una estructura homogeneizada a ser utilizada posteriormente por el predictor. 
En particular, Financing no se une actividad a actividad: se agrega a recipient + year y luego se pega por ISO3. Para evitar duplicados se conserva un único project_id.



## Procesamiento de limpieza



Tras realizar un análisis de las fuentes, se determinó que únicamente cuatro archivos eran necesarios para la construcción del dataset final utilizado por el sistema de predicción.
Para construir la tabla de proyectos evaluados se utilizó:

- PPD2_Jan_21_2022.csv

Para construir el contexto del país se utilizaron:
- SDGData.csv
- SDGCountry.csv
  
Para construir el entorno de cooperación (ODS 17) se utilizó:
- FinancingtotheSDGsDataset_v1.0.csv
  
Durante la lectura de los datasets, previo a construir el dataset interno del sistema, se llevó a cabo el preprocesamiento. Esta fase permitió organizar la información que posteriormente sería utilizada
en la parte del procesamiento secuencial y concurrente. No se imputó ningún hueco: lo inválido pasa a nulo o se descarta la fila.

Se implementó la función de limpieza como parte del preprocesamiento, en la que se aplica la normalización de texto y de códigos de país. Con ello se eliminaron espacios no separadores, textos vacíos, comillas envolventes y filas de parseo corrupto. Se homologaron nombres de país a ISO3 y se castearon a float las columnas goal_1 … goal_17 de Financing, que venían como texto (“NA”, negativos). De este modo también se procedía a homogeneizar los registros extraídos de orígenes diferentes.
Por ejemplo: “Viet Nam”, “Vietnam” y el código VNM se homologaron al metavalor ISO3 VNM. Côte d’Ivoire, Congo y West Bank and Gaza quedan como CIV, COG y PSE. El cruce se hace por código y no por cadena libre.

De igual manera, se llevó a cabo el proceso de corrección de la unidad de duración. En el PPD la columna project_duration está en días (mediana cruda ≈ 2 177, equivalentes a unos 6 años). El sistema la convierte a duration_years; si el valor es negativo, nulo de origen, ambiguo entre 1 y 40, o mayor a 25 años tras convertir, se anula. Tras la conversión, duration_years tiene mediana 6.1 años (mínimo 0.11, máximo 23.1).
También se incorporaron validaciones de campos numéricos para evitar inconsistencias durante el procesamiento de los datos. Los ratings fuera de 1–6 no se conservan como etiqueta. El compromiso no positivo pasa a nulo (621 celdas). Los porcentajes se exigen en 0–100. Los 869 decommitments de Financing (compromiso < 0, 0.069%) no se recortaron a cero: son cancelaciones reales y se marcaron. Los 44 agregados del Banco Mundial (World, South Asia, Arab World) no se tratan como país.


**Filas eliminadas del PPD**

| **Paso** | **Filas** | **Quedan** |
|---|---:|---:|
| PPD crudo | 21 200 | 21 200 |
| Sin `six_overall_rating` (sin etiqueta) | 513 | 20 687 |
| Con nota, pero sin año de inicio válido | 64 | 20 623 |
| Total eliminado del PPD (2.72%) | 577 | 20 623 |


Además había 2 filas de donante ilegible y 2 filas con el mismo project_id (se conservó una). 
Esos casos ya quedan cubiertos en los cortes anteriores. No se eliminan las aproximadamente 480 columnas de ratings de
agencia por filas: simplemente no se leen, porque emplearlas como input sería fuga de información.


Las 1 252 036 filas de Financing y las 106 488 de SDGData no se recortan como tabla de entrenamiento: se agregan y se pegan. El parquet conserva 
las 20 623 filas sanitizadas. fila_ok identifica el subconjunto usable (país real e inicio entre 1990 y 2017): 13 915 filas. Las otras 6 708 no se borran, para
poder describir sesgo y cobertura. De las usable, 8 389 caen en 2000–2013, donde pueden cruzar las tres fuentes. La tasa de éxito es 73.7% en la tabla sanitizada y 75.4% dentro de fila_ok.
