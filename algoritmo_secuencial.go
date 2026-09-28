package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"

	"randomforest-go/rfcore"
)

func main() {

	inicio := time.Now() // Para medir el tiempo

	path := "model_table.parquet" // cambiar si la dirección es diferente
	if len(os.Args) > 1 {
		path = os.Args[1]
	}
	nTrees := 50
	if len(os.Args) > 2 {
		if v, err := strconv.Atoi(os.Args[2]); err == nil {
			nTrees = v
		}
	}

	fmt.Println("DataLoader: cargando y validando dataset...")
	ds, err := rfcore.LoadDataset(path)
	if err != nil {
		fmt.Println("Error cargando dataset:", err)
		os.Exit(1)
	}
	fmt.Printf("DataLoader: dataset listo (fila_ok=true) -> %d filas válidas, %d features (%v)\n",
		len(ds.X), len(ds.FeatureNames), ds.FeatureNames)

	trainX, testX, trainY, testY := rfcore.TrainTestSplit(ds, 0.2, 42)
	fmt.Printf("DataLoader: train=%d, test=%d\n", len(trainX), len(testX))

	params := rfcore.DefaultTreeParams(len(ds.FeatureNames))

	start := time.Now()
	trees := make([]*rfcore.Node, nTrees)

	// entrenamiento SECUENCIAL: un árbol detrás de otro
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < nTrees; i++ {
		bx, by := rfcore.BootstrapSample(trainX, trainY, rng)
		trees[i] = rfcore.BuildTree(bx, by, params, 0, rng)
		fmt.Printf("Tree[%d]: entrenado con muestra bootstrap (%d filas)\n", i, len(bx))
	}
	elapsed := time.Since(start)

	// Aggregator: predicción del bosque por mayoría de votos
	predY := make([]int, len(testX))
	for i, row := range testX {
		votes := make([]int, nTrees)
		for t, tree := range trees {
			votes[t] = rfcore.Predict(tree, row)
		}
		predY[i] = rfcore.MajorityVote(votes)
	}

	acc := rfcore.Accuracy(testY, predY)
	fmt.Printf("Aggregator: accuracy sobre test set = %.4f (%d árboles, %v)\n", acc, nTrees, elapsed)
	duracion := time.Since(inicio)
	fmt.Printf("Tiempo que se usó: %4f segundos \n", duracion.Seconds())
}
