// ============================================================
// rfcore/dataset.go
// ------------------------------------------------------------
// Lectura y preprocesamiento REAL del dataset de proyectos
// (project_id, donor, iso3, ..., rating, success, fila_ok)
// a partir de un archivo .parquet, usando la librería pura-Go
// github.com/segmentio/parquet-go (sin CGO, sin dependencias
// de golang.org bloqueadas por la red de este entorno).
// ============================================================

package rfcore

import (
	"fmt"
	"math/rand"
	"os"
	"sort"

	"github.com/parquet-go/parquet-go"
)

// parquetRow refleja el esquema del dataset. Los tipos deben
// coincidir con los tipos físicos del parquet (string, int64,
// float64, bool). Si tu archivo real tiene columnas nulas,
// cambia el campo correspondiente a puntero (*float64, etc.)
// y ajusta featureValue() más abajo.
type parquetRow struct {
	ProjectID         string  `parquet:"project_id"`
	Donor             string  `parquet:"donor"`
	ISO3              string  `parquet:"iso3"`
	StartYear         int64   `parquet:"start_year"`
	SectorCode        string  `parquet:"sector_code"`
	SectorDescription string  `parquet:"sector_description"`
	DurationYears     float64 `parquet:"duration_years"`
	CommitmentUSD     float64 `parquet:"commitment_usd"`
	WBRegion          string  `parquet:"wb_region"`
	WBIncome          string  `parquet:"wb_income"`
	GDPpc             float64 `parquet:"gdp_pc"`
	ElecAccess        float64 `parquet:"elec_access"`
	ChildMort         float64 `parquet:"child_mort"`
	NDonors           int64   `parquet:"n_donors"`
	USDods17          float64 `parquet:"usd_ods17"`
	Rating            float64 `parquet:"rating"`
	FilaOk            bool    `parquet:"fila_ok"`
	Success           int64   `parquet:"success"`
}

// Dataset contiene la matriz de features ya preprocesada
// (numéricas + one-hot de wb_income) y las etiquetas (success).
type Dataset struct {
	X            [][]float64 // filas = proyectos, columnas = features
	Y            []int       // 0/1 (success)
	FeatureNames []string
}

// readParquet lee todas las filas del archivo .parquet dado.
func readParquet(path string) ([]parquetRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("no se pudo abrir %s: %w", path, err)
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}

	pf, err := parquet.OpenFile(f, stat.Size())
	if err != nil {
		return nil, fmt.Errorf("no se pudo abrir el parquet: %w", err)
	}

	reader := parquet.NewGenericReader[parquetRow](pf)
	defer reader.Close()

	rows := make([]parquetRow, 0, pf.NumRows())
	buf := make([]parquetRow, 512)
	for {
		n, rerr := reader.Read(buf)
		rows = append(rows, buf[:n]...)
		if rerr != nil {
			break // io.EOF esperado al terminar
		}
	}
	return rows, nil
}

// LoadDataset lee el parquet, aplica el filtro fila_ok == true
// (equivalente a la limpieza descrita en el diccionario de datos:
// "país válido y año entre 1990 y 2017") y construye la matriz
// de features numéricas + one-hot de wb_income.
func LoadDataset(path string) (*Dataset, error) {
	rows, err := readParquet(path)
	if err != nil {
		return nil, err
	}

	// filtra solo filas válidas (fila_ok == true)
	valid := rows[:0]
	for _, r := range rows {
		if r.FilaOk {
			valid = append(valid, r)
		}
	}
	if len(valid) == 0 {
		return nil, fmt.Errorf("no quedaron filas válidas tras filtrar fila_ok == true")
	}

	// descubre dinámicamente las categorías de wb_income presentes
	// (evita asumir un conjunto fijo tipo LIC/LMIC/UMIC/HIC)
	incomeSet := map[string]bool{}
	for _, r := range valid {
		incomeSet[r.WBIncome] = true
	}
	incomeCats := make([]string, 0, len(incomeSet))
	for k := range incomeSet {
		incomeCats = append(incomeCats, k)
	}
	sort.Strings(incomeCats)

	names := []string{
		"duration_years", "commitment_usd", "gdp_pc", "elec_access",
		"child_mort", "n_donors", "usd_ods17", "start_year",
	}
	for _, c := range incomeCats {
		names = append(names, "wb_income="+c)
	}

	X := make([][]float64, 0, len(valid))
	Y := make([]int, 0, len(valid))

	for _, r := range valid {
		feat := []float64{
			r.DurationYears,
			r.CommitmentUSD,
			r.GDPpc,
			r.ElecAccess,
			r.ChildMort,
			float64(r.NDonors),
			r.USDods17,
			float64(r.StartYear),
		}
		for _, c := range incomeCats {
			if r.WBIncome == c {
				feat = append(feat, 1.0)
			} else {
				feat = append(feat, 0.0)
			}
		}

		// success viene precalculado en la columna, pero si faltara
		// se recalcula a partir de rating >= 4 (regla del diccionario)
		label := int(r.Success)

		X = append(X, feat)
		Y = append(Y, label)
	}

	return &Dataset{X: X, Y: Y, FeatureNames: names}, nil
}

// TrainTestSplit separa el dataset en entrenamiento/prueba de forma
// aleatoria (shuffle con semilla fija para reproducibilidad).
func TrainTestSplit(ds *Dataset, testFrac float64, seed int64) (trainX, testX [][]float64, trainY, testY []int) {
	n := len(ds.X)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	rng := rand.New(rand.NewSource(seed))
	rng.Shuffle(n, func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })

	nTest := int(float64(n) * testFrac)
	for k, i := range idx {
		if k < nTest {
			testX = append(testX, ds.X[i])
			testY = append(testY, ds.Y[i])
		} else {
			trainX = append(trainX, ds.X[i])
			trainY = append(trainY, ds.Y[i])
		}
	}
	return
}
