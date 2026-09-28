package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"omega/rfcore"
)

type medicion struct {
	modo    string
	workers int
	tiempos []float64
	allocMB float64
}

func main() {
	data := flag.String("data", "datasets/csv/model_table.csv", "CSV exportado con exportar_csv.py")
	nTrees := flag.Int("trees", 50, "número de árboles")
	reps := flag.Int("reps", 10, "ejecuciones por configuración")
	trim := flag.Float64("trim", 0.1, "fracción recortada en cada extremo")
	workersFlag := flag.String("workers", "1,2,4,8", "lista de workers a probar")
	seed := flag.Int64("seed", 42, "semilla")
	out := flag.String("out", "resultados/benchmark.csv", "CSV de salida")
	flag.Parse()

	var workerList []int
	for _, s := range strings.Split(*workersFlag, ",") {
		w, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || w < 1 {
			fmt.Fprintf(os.Stderr, "workers inválido: %q\n", s)
			os.Exit(1)
		}
		workerList = append(workerList, w)
	}

	ds, err := rfcore.LoadCSV(*data, false)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
	split := rfcore.TrainTestSplit(ds, 0.2, *seed)
	params := rfcore.DefaultTreeParams(len(ds.FeatureNames))
	fmt.Printf("CPUs=%d GOMAXPROCS=%d | train=%d filas, %d features | %d árboles, %d reps, recorte %.0f%%\n\n",
		runtime.NumCPU(), runtime.GOMAXPROCS(0), len(split.TrainY), len(ds.FeatureNames), *nTrees, *reps, *trim*100)

	ref := rfcore.TrainSequential(split.TrainX, split.TrainY, *nTrees, params, *seed).PredictAll(split.TestX)

	medir := func(modo string, w int, train func() *rfcore.Forest) medicion {
		m := medicion{modo: modo, workers: w}
		var allocTotal uint64
		for r := 0; r < *reps; r++ {
			runtime.GC()
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			start := time.Now()
			forest := train()
			m.tiempos = append(m.tiempos, time.Since(start).Seconds())
			runtime.ReadMemStats(&after)
			allocTotal += after.TotalAlloc - before.TotalAlloc
			if r == 0 && !mismasPredicciones(ref, forest.PredictAll(split.TestX)) {
				fmt.Fprintf(os.Stderr, "ERROR: %s con %d workers no reproduce el bosque secuencial\n", modo, w)
				os.Exit(1)
			}
		}
		m.allocMB = float64(allocTotal) / float64(*reps) / (1 << 20)
		return m
	}

	resultados := []medicion{medir("secuencial", 1, func() *rfcore.Forest {
		return rfcore.TrainSequential(split.TrainX, split.TrainY, *nTrees, params, *seed)
	})}
	for _, w := range workerList {
		w := w
		resultados = append(resultados, medir("concurrente", w, func() *rfcore.Forest {
			return rfcore.TrainConcurrent(split.TrainX, split.TrainY, *nTrees, w, params, *seed)
		}))
	}

	tSeq := rfcore.TrimmedMean(resultados[0].tiempos, *trim)
	fmt.Printf("%-12s %7s %12s %9s %9s %9s %9s %10s %9s\n",
		"modo", "workers", "media_rec(s)", "media(s)", "desv(s)", "min(s)", "max(s)", "speedup", "eficien.")
	filas := [][]string{{"modo", "workers", "media_recortada_s", "media_s", "desv_s", "min_s", "max_s", "speedup", "eficiencia", "memoria_asignada_mb"}}
	for _, m := range resultados {
		tm := rfcore.TrimmedMean(m.tiempos, *trim)
		lo, hi := rfcore.MinMax(m.tiempos)
		speedup := tSeq / tm
		eficiencia := speedup / float64(m.workers)
		fmt.Printf("%-12s %7d %12.4f %9.4f %9.4f %9.4f %9.4f %9.2fx %9.2f\n",
			m.modo, m.workers, tm, rfcore.Mean(m.tiempos), rfcore.StdDev(m.tiempos), lo, hi, speedup, eficiencia)
		filas = append(filas, []string{
			m.modo, strconv.Itoa(m.workers), f4(tm), f4(rfcore.Mean(m.tiempos)), f4(rfcore.StdDev(m.tiempos)),
			f4(lo), f4(hi), f4(speedup), f4(eficiencia), f4(m.allocMB),
		})
	}
	fmt.Println("\nTodas las versiones concurrentes reproducen exactamente el bosque secuencial.")

	if err := guardarCSV(*out, filas); err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
	fmt.Println("Resultados guardados en", *out)
}

func mismasPredicciones(a, b []int) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return len(a) == len(b)
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
