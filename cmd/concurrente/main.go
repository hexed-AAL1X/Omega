package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"time"

	"omega/rfcore"
)

func main() {
	data := flag.String("data", "datasets/csv/model_table.csv", "CSV exportado con exportar_csv.py")
	nTrees := flag.Int("trees", 50, "número de árboles")
	workers := flag.Int("workers", runtime.NumCPU(), "goroutines del worker pool")
	seed := flag.Int64("seed", 42, "semilla")
	ventana := flag.Bool("ventana", false, "usar solo proyectos que inician en 2000-2013")
	flag.Parse()

	t0 := time.Now()
	ds, err := rfcore.LoadCSV(*data, *ventana)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
	split := rfcore.TrainTestSplit(ds, 0.2, *seed)
	fmt.Printf("Datos: %d filas, %d features | train=%d test=%d | carga %v\n",
		len(ds.X), len(ds.FeatureNames), len(split.TrainY), len(split.TestY), time.Since(t0).Round(time.Millisecond))

	params := rfcore.DefaultTreeParams(len(ds.FeatureNames))
	start := time.Now()
	forest := rfcore.TrainConcurrent(split.TrainX, split.TrainY, *nTrees, *workers, params, *seed)
	elapsed := time.Since(start)

	m := rfcore.Evaluate(split.TestY, forest.PredictAll(split.TestX))
	fmt.Printf("Concurrente: %d árboles con %d workers en %v (CPUs: %d)\n",
		*nTrees, *workers, elapsed.Round(time.Millisecond), runtime.NumCPU())
	fmt.Println("Test:", m)
}
