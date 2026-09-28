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

	"omega/pipeline"
	"omega/recursos"
	"omega/rfcore"
)

type fila struct {
	modo     string
	workers  int
	frac     float64
	filas    int
	medidas  []recursos.Medicion
	tiempos  []float64
	mediaRec float64
}

func main() {
	data := flag.String("data", "datasets/financing_sdgs/FinancingtotheSDGsDataset_v1.0.csv", "CSV de Financing to the SDGs")
	reps := flag.Int("reps", 10, "ejecuciones por configuración")
	trim := flag.Float64("trim", 0.1, "fracción recortada en cada extremo")
	workersFlag := flag.String("workers", "1,2,4,8,16", "lista de workers")
	fracsFlag := flag.String("fracs", "0.25,0.5,0.75,1", "fracciones del archivo para escalabilidad")
	bloquesPorWorker := flag.Int("bloques", 8, "bloques por worker")
	out := flag.String("out", "resultados", "carpeta de salida")
	flag.Parse()

	workersList := enteros(*workersFlag)
	fracs := decimales(*fracsFlag)

	contenido, err := os.ReadFile(*data)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err, "(corre python download_datasets.py)")
		os.Exit(1)
	}
	ref, err := pipeline.Secuencial(contenido)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
	fmt.Printf("Financing: %d MB | %d filas | %d país-año | %d inválidas | CPUs=%d\n\n",
		len(contenido)>>20, ref.Filas, len(ref.Grupos), ref.Invalidas, runtime.NumCPU())

	medir := func(modo string, w int, frac float64) fila {
		parte := pipeline.Recortar(contenido, frac)
		f := fila{modo: modo, workers: w, frac: frac}
		var base pipeline.Resultado
		for r := 0; r < *reps; r++ {
			var res pipeline.Resultado
			var e error
			m := recursos.Medir(func() {
				if modo == "secuencial" {
					res, e = pipeline.Secuencial(parte)
				} else {
					res, e = pipeline.Concurrente(parte, w, w**bloquesPorWorker)
				}
			})
			if e != nil {
				fmt.Fprintln(os.Stderr, "Error:", e)
				os.Exit(1)
			}
			if r == 0 {
				base, _ = pipeline.Secuencial(parte)
				if err := pipeline.Iguales(base, res); err != nil {
					fmt.Fprintf(os.Stderr, "ERROR: %s con %d workers no coincide: %v\n", modo, w, err)
					os.Exit(1)
				}
			}
			f.filas = res.Filas
			f.medidas = append(f.medidas, m)
			f.tiempos = append(f.tiempos, m.Segundos)
		}
		f.mediaRec = rfcore.TrimmedMean(f.tiempos, *trim)
		return f
	}

	porWorkers := []fila{medir("secuencial", 1, 1)}
	for _, w := range workersList {
		porWorkers = append(porWorkers, medir("concurrente", w, 1))
	}
	imprimir("Speedup por número de workers (archivo completo)", porWorkers, porWorkers[0].mediaRec)

	wMax := runtime.NumCPU()
	var porTamano []fila
	for _, fr := range fracs {
		s := medir("secuencial", 1, fr)
		c := medir("concurrente", wMax, fr)
		porTamano = append(porTamano, s, c)
	}
	fmt.Printf("\nEscalabilidad por tamaño de datos (concurrente con %d workers)\n", wMax)
	fmt.Printf("%6s %10s %12s %12s %9s\n", "frac", "filas", "secuencial", "concurrente", "speedup")
	for i := 0; i < len(porTamano); i += 2 {
		s, c := porTamano[i], porTamano[i+1]
		fmt.Printf("%6.2f %10d %11.4fs %11.4fs %8.2fx\n", s.frac, s.filas, s.mediaRec, c.mediaRec, s.mediaRec/c.mediaRec)
	}

	if err := guardar(filepath.Join(*out, "pipeline_workers.csv"), porWorkers, func(f fila) float64 { return porWorkers[0].mediaRec }); err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
	seqPorFrac := map[float64]float64{}
	for _, f := range porTamano {
		if f.modo == "secuencial" {
			seqPorFrac[f.frac] = f.mediaRec
		}
	}
	if err := guardar(filepath.Join(*out, "pipeline_tamano.csv"), porTamano, func(f fila) float64 { return seqPorFrac[f.frac] }); err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
	fmt.Println("\nTodas las ejecuciones concurrentes coinciden con la secuencial.")
	fmt.Println("Resultados en", filepath.Join(*out, "pipeline_workers.csv"), "y", filepath.Join(*out, "pipeline_tamano.csv"))
}

func imprimir(titulo string, filas []fila, tSeq float64) {
	fmt.Println(titulo)
	fmt.Printf("%-12s %7s %12s %9s %9s %9s %8s %9s\n", "modo", "workers", "media_rec(s)", "desv(s)", "speedup", "eficien.", "CPU %", "heap MB")
	for _, f := range filas {
		sp := tSeq / f.mediaRec
		fmt.Printf("%-12s %7d %12.4f %9.4f %8.2fx %9.2f %7.1f%% %9.1f\n",
			f.modo, f.workers, f.mediaRec, rfcore.StdDev(f.tiempos), sp, sp/float64(f.workers),
			100*promedio(f.medidas, func(m recursos.Medicion) float64 { return m.UsoCPU }),
			promedio(f.medidas, func(m recursos.Medicion) float64 { return m.HeapPicoMB }))
	}
}

func promedio(ms []recursos.Medicion, g func(recursos.Medicion) float64) float64 {
	vals := make([]float64, len(ms))
	for i, m := range ms {
		vals[i] = g(m)
	}
	return rfcore.Mean(vals)
}

func guardar(path string, filas []fila, tSeq func(fila) float64) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	w.Write([]string{"modo", "workers", "fraccion", "filas", "media_recortada_s", "media_s", "desv_s", "min_s", "max_s",
		"speedup", "eficiencia", "cpu_seg", "uso_cpu_pct", "heap_pico_mb", "memoria_asignada_mb"})
	for _, fl := range filas {
		lo, hi := rfcore.MinMax(fl.tiempos)
		sp := tSeq(fl) / fl.mediaRec
		w.Write([]string{
			fl.modo, strconv.Itoa(fl.workers), f4(fl.frac), strconv.Itoa(fl.filas),
			f4(fl.mediaRec), f4(rfcore.Mean(fl.tiempos)), f4(rfcore.StdDev(fl.tiempos)), f4(lo), f4(hi),
			f4(sp), f4(sp / float64(fl.workers)),
			f4(promedio(fl.medidas, func(m recursos.Medicion) float64 { return m.CPUSegundos })),
			f4(100 * promedio(fl.medidas, func(m recursos.Medicion) float64 { return m.UsoCPU })),
			f4(promedio(fl.medidas, func(m recursos.Medicion) float64 { return m.HeapPicoMB })),
			f4(promedio(fl.medidas, func(m recursos.Medicion) float64 { return m.AllocMB })),
		})
	}
	w.Flush()
	return w.Error()
}

func f4(v float64) string { return strconv.FormatFloat(v, 'f', 4, 64) }

func enteros(s string) []int {
	var out []int
	for _, p := range strings.Split(s, ",") {
		v, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil || v < 1 {
			fmt.Fprintf(os.Stderr, "valor inválido: %q\n", p)
			os.Exit(1)
		}
		out = append(out, v)
	}
	return out
}

func decimales(s string) []float64 {
	var out []float64
	for _, p := range strings.Split(s, ",") {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil || v <= 0 || v > 1 {
			fmt.Fprintf(os.Stderr, "fracción inválida: %q\n", p)
			os.Exit(1)
		}
		out = append(out, v)
	}
	return out
}
