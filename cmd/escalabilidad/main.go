package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"omega/recursos"
	"omega/rfcore"
)

type config struct {
	experimento string
	fraccion    float64
	arboles     int
}

func main() {
	data := flag.String("data", "datasets/csv/model_table.csv", "CSV exportado con exportar_csv.py")
	reps := flag.Int("reps", 5, "ejecuciones por configuración")
	trim := flag.Float64("trim", 0.2, "fracción recortada en cada extremo")
	fracsFlag := flag.String("fracs", "0.25,0.5,0.75,1", "fracciones del conjunto de entrenamiento")
	arbolesFlag := flag.String("trees", "25,50,100,200", "cantidades de árboles")
	arbolesBase := flag.Int("trees-base", 50, "árboles en el experimento de tamaño de datos")
	workers := flag.Int("workers", runtime.NumCPU(), "workers de la versión concurrente")
	seed := flag.Int64("seed", 42, "semilla")
	out := flag.String("out", "resultados/escalabilidad.csv", "CSV de salida")
	flag.Parse()

	fracs, err := listaFloat(*fracsFlag)
	salirSi(err)
	arboles, err := listaInt(*arbolesFlag)
	salirSi(err)

	ds, err := rfcore.LoadCSV(*data, false)
	salirSi(err)
	split := rfcore.TrainTestSplit(ds, 0.2, *seed)
	params := rfcore.DefaultTreeParams(len(ds.FeatureNames))

	var configs []config
	for _, f := range fracs {
		configs = append(configs, config{"tamano_datos", f, *arbolesBase})
	}
	for _, a := range arboles {
		configs = append(configs, config{"numero_arboles", 1, a})
	}

	fmt.Printf("CPUs=%d | train completo=%d filas | test=%d filas | workers=%d | reps=%d\n\n",
		runtime.NumCPU(), len(split.TrainY), len(split.TestY), *workers, *reps)
	fmt.Printf("%-15s %6s %7s %7s %11s %11s %8s %8s %8s %9s\n",
		"experimento", "frac", "filas", "árboles", "secuen.(s)", "concur.(s)", "speedup", "CPU%seq", "CPU%con", "bal.acc")

	filas := [][]string{{"experimento", "fraccion", "filas_train", "arboles", "modo", "workers",
		"media_recortada_s", "media_s", "desv_s", "speedup", "eficiencia", "uso_cpu_pct",
		"heap_pico_mb", "memoria_asignada_mb", "balanced_accuracy"}}

	for _, c := range configs {
		X, y := submuestra(split.TrainX, split.TrainY, c.fraccion, *seed)

		var ref []int
		correr := func(modo string, w int) ([]float64, recursos.Medicion, rfcore.Metrics) {
			var tiempos []float64
			var acum recursos.Medicion
			var met rfcore.Metrics
			for r := 0; r < *reps; r++ {
				var bosque *rfcore.Forest
				m := recursos.Medir(func() {
					if w == 1 && modo == "secuencial" {
						bosque = rfcore.TrainSequential(X, y, c.arboles, params, *seed)
					} else {
						bosque = rfcore.TrainConcurrent(X, y, c.arboles, w, params, *seed)
					}
				})
				tiempos = append(tiempos, m.Segundos)
				acum.UsoCPU += m.UsoCPU
				acum.AllocMB += m.AllocMB
				if m.HeapPicoMB > acum.HeapPicoMB {
					acum.HeapPicoMB = m.HeapPicoMB
				}
				if r == 0 {
					pred := bosque.PredictAll(split.TestX)
					met = rfcore.Evaluate(split.TestY, pred)
					if ref == nil {
						ref = pred
					} else if !iguales(ref, pred) {
						fmt.Fprintf(os.Stderr, "ERROR: %s no reproduce el bosque secuencial (%+v)\n", modo, c)
						os.Exit(1)
					}
				}
			}
			acum.UsoCPU /= float64(*reps)
			acum.AllocMB /= float64(*reps)
			return tiempos, acum, met
		}

		tSeq, mSeq, met := correr("secuencial", 1)
		tCon, mCon, _ := correr("concurrente", *workers)
		seq := rfcore.TrimmedMean(tSeq, *trim)
		con := rfcore.TrimmedMean(tCon, *trim)
		speedup := seq / con

		fmt.Printf("%-15s %6.2f %7d %7d %11.4f %11.4f %7.2fx %7.1f%% %7.1f%% %9.3f\n",
			c.experimento, c.fraccion, len(y), c.arboles, seq, con, speedup,
			mSeq.UsoCPU*100, mCon.UsoCPU*100, met.BalancedAccuracy)

		for _, fila := range []struct {
			modo    string
			w       int
			tiempos []float64
			med     recursos.Medicion
			sp      float64
		}{{"secuencial", 1, tSeq, mSeq, 1}, {"concurrente", *workers, tCon, mCon, speedup}} {
			filas = append(filas, []string{
				c.experimento, f4(c.fraccion), strconv.Itoa(len(y)), strconv.Itoa(c.arboles), fila.modo,
				strconv.Itoa(fila.w), f4(rfcore.TrimmedMean(fila.tiempos, *trim)), f4(rfcore.Mean(fila.tiempos)),
				f4(rfcore.StdDev(fila.tiempos)), f4(fila.sp), f4(fila.sp / float64(fila.w)),
				f4(fila.med.UsoCPU * 100), f4(fila.med.HeapPicoMB), f4(fila.med.AllocMB), f4(met.BalancedAccuracy),
			})
		}
	}

	fmt.Println("\nLas versiones concurrentes reproducen exactamente el bosque secuencial en todas las configuraciones.")
	salirSi(guardarCSV(*out, filas))
	fmt.Println("Resultados guardados en", *out)
}

func submuestra(X [][]float64, y []int, frac float64, seed int64) ([][]float64, []int) {
	if frac >= 1 {
		return X, y
	}
	idx := rand.New(rand.NewSource(seed)).Perm(len(y))
	n := int(float64(len(y)) * frac)
	sx := make([][]float64, n)
	sy := make([]int, n)
	for i := 0; i < n; i++ {
		sx[i] = X[idx[i]]
		sy[i] = y[idx[i]]
	}
	return sx, sy
}

func iguales(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func listaFloat(s string) ([]float64, error) {
	var out []float64
	for _, p := range strings.Split(s, ",") {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil || v <= 0 || v > 1 {
			return nil, fmt.Errorf("fracción inválida: %q", p)
		}
		out = append(out, v)
	}
	return out, nil
}

func listaInt(s string) ([]int, error) {
	var out []int
	for _, p := range strings.Split(s, ",") {
		v, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil || v < 1 {
			return nil, fmt.Errorf("valor inválido: %q", p)
		}
		out = append(out, v)
	}
	return out, nil
}

func salirSi(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}

func f4(v float64) string { return strconv.FormatFloat(v, 'f', 4, 64) }

func guardarCSV(path string, filas [][]string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	if err := w.WriteAll(filas); err != nil {
		return err
	}
	return w.Error()
}
