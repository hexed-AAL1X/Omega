package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"

	"randomforest-go/rfcore"
)

// sharedState agrupa lo que las goroutines Tree escriben,
// protegido por mutex — igual que en el modelo Promela.
type sharedState struct {
	mu    sync.Mutex
	trees []*rfcore.Node
	done  int
}

func treeWorker(id int, trainX [][]float64, trainY []int, params rfcore.TreeParams,
	st *sharedState, wg *sync.WaitGroup) {
	defer wg.Done()

	// cada goroutine tiene su propio generador aleatorio (independiente
	// y libre de condiciones de carrera sobre el estado del rng)
	rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(id)))

	bx, by := rfcore.BootstrapSample(trainX, trainY, rng)
	tree := rfcore.BuildTree(bx, by, params, 0, rng)

	// sección crítica: publicar el árbol entrenado (equivalente al
	// atomic{} que actualizaba votes[]/finished en Promela)
	st.mu.Lock()
	st.trees[id] = tree
	st.done++
	fmt.Printf("Tree[%d]: entrenado con muestra bootstrap (%d filas, done=%d)\n", id, len(bx), st.done)
	st.mu.Unlock()
}

func main() {

	inicio := time.Now() // Para medir el tiempo

	path := "model_table.parquet"
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

	st := &sharedState{trees: make([]*rfcore.Node, nTrees)}
	var wg sync.WaitGroup
	wg.Add(nTrees)

	start := time.Now()
	for i := 0; i < nTrees; i++ {
		go treeWorker(i, trainX, trainY, params, st, &wg)
	}

	// barrera: el Aggregator no avanza hasta que finished == NTREES
	wg.Wait()
	elapsed := time.Since(start)

	// Aggregator: predicción del bosque por mayoría de votos
	predY := make([]int, len(testX))
	for i, row := range testX {
		votes := make([]int, nTrees)
		for t, tree := range st.trees {
			votes[t] = rfcore.Predict(tree, row)
		}
		predY[i] = rfcore.MajorityVote(votes)
	}

	acc := rfcore.Accuracy(testY, predY)
	fmt.Printf("Aggregator: accuracy sobre test set = %.4f (%d árboles, %v)\n", acc, nTrees, elapsed)
	duracion := time.Since(inicio)
	fmt.Printf("Tiempo que se usó: %4f segundos \n", duracion.Seconds())
}
