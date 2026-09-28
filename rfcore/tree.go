// ============================================================
// rfcore/tree.go
// ------------------------------------------------------------
// Implementación real de un árbol de decisión CART (impureza
// Gini, split binario por umbral) más las utilidades de
// Random Forest: muestreo bootstrap, submuestreo de features
// por nodo (feature bagging) y votación por mayoría.
//
// Esto reemplaza el "voto no determinista" del modelo Promela
// original por un entrenamiento real sobre los datos.
// ============================================================

package rfcore

import "math/rand"

// TreeParams controla la complejidad de cada árbol.
type TreeParams struct {
	MaxDepth        int // profundidad máxima
	MinSamplesSplit int // mínimo de muestras para intentar dividir un nodo
	MaxFeatures     int // features consideradas por split (bagging de features)
}

// DefaultTreeParams da valores razonables por defecto.
func DefaultTreeParams(nFeatures int) TreeParams {
	mf := nFeatures
	if mf > 3 {
		mf = 3 // ~sqrt(nFeatures) para un dataset de ~11 features
	}
	return TreeParams{MaxDepth: 8, MinSamplesSplit: 10, MaxFeatures: mf}
}

// Node es un nodo del árbol: interno (con split) u hoja (con predicción).
type Node struct {
	IsLeaf     bool
	Prediction int // válido si IsLeaf

	FeatureIdx int
	Threshold  float64
	Left       *Node
	Right      *Node
}

// gini calcula la impureza Gini de un conjunto de etiquetas.
func gini(y []int) float64 {
	if len(y) == 0 {
		return 0
	}
	var ones int
	for _, v := range y {
		ones += v
	}
	p1 := float64(ones) / float64(len(y))
	p0 := 1 - p1
	return 1 - p1*p1 - p0*p0
}

// majorityLabel devuelve la clase mayoritaria de un conjunto de etiquetas.
func majorityLabel(y []int) int {
	ones := 0
	for _, v := range y {
		ones += v
	}
	if ones*2 >= len(y) {
		return 1
	}
	return 0
}

// bestSplit busca, sobre un subconjunto aleatorio de features
// (feature bagging, igual que scikit-learn RandomForest), el
// (feature, umbral) que minimiza la impureza Gini ponderada.
func bestSplit(X [][]float64, y []int, candidateFeatures []int) (featIdx int, threshold float64, found bool) {
	n := len(y)
	if n == 0 {
		return 0, 0, false
	}
	baseGini := gini(y)
	bestGain := 0.0
	found = false

	for _, f := range candidateFeatures {
		// candidatos de umbral: valores únicos ordenados de esa feature
		vals := make([]float64, n)
		for i := range X {
			vals[i] = X[i][f]
		}
		sortedUnique := uniqueSorted(vals)
		for i := 0; i+1 < len(sortedUnique); i++ {
			thr := (sortedUnique[i] + sortedUnique[i+1]) / 2

			var leftY, rightY []int
			for i2, row := range X {
				if row[f] <= thr {
					leftY = append(leftY, y[i2])
				} else {
					rightY = append(rightY, y[i2])
				}
			}
			if len(leftY) == 0 || len(rightY) == 0 {
				continue
			}
			wl := float64(len(leftY)) / float64(n)
			wr := float64(len(rightY)) / float64(n)
			childGini := wl*gini(leftY) + wr*gini(rightY)
			gain := baseGini - childGini
			if gain > bestGain {
				bestGain = gain
				featIdx = f
				threshold = thr
				found = true
			}
		}
	}
	return
}

func uniqueSorted(vals []float64) []float64 {
	cp := append([]float64(nil), vals...)
	// insertion sort simple (los datasets por nodo son pequeños)
	for i := 1; i < len(cp); i++ {
		for j := i; j > 0 && cp[j-1] > cp[j]; j-- {
			cp[j-1], cp[j] = cp[j], cp[j-1]
		}
	}
	out := cp[:0]
	var last float64
	for i, v := range cp {
		if i == 0 || v != last {
			out = append(out, v)
			last = v
		}
	}
	return out
}

// sampleFeatures elige aleatoriamente k índices de features
// (sin repetición) de entre 0..nFeatures-1.
func sampleFeatures(nFeatures, k int, rng *rand.Rand) []int {
	if k >= nFeatures {
		out := make([]int, nFeatures)
		for i := range out {
			out[i] = i
		}
		return out
	}
	perm := rng.Perm(nFeatures)
	return perm[:k]
}

// BuildTree entrena recursivamente un árbol CART real sobre (X, y).
func BuildTree(X [][]float64, y []int, params TreeParams, depth int, rng *rand.Rand) *Node {
	if len(y) == 0 {
		return &Node{IsLeaf: true, Prediction: 0}
	}
	// condiciones de parada: pureza, profundidad máxima o muy pocas muestras
	if depth >= params.MaxDepth || len(y) < params.MinSamplesSplit || gini(y) == 0 {
		return &Node{IsLeaf: true, Prediction: majorityLabel(y)}
	}

	nFeatures := len(X[0])
	candidates := sampleFeatures(nFeatures, params.MaxFeatures, rng)

	f, thr, ok := bestSplit(X, y, candidates)
	if !ok {
		return &Node{IsLeaf: true, Prediction: majorityLabel(y)}
	}

	var leftX, rightX [][]float64
	var leftY, rightY []int
	for i, row := range X {
		if row[f] <= thr {
			leftX = append(leftX, row)
			leftY = append(leftY, y[i])
		} else {
			rightX = append(rightX, row)
			rightY = append(rightY, y[i])
		}
	}

	node := &Node{FeatureIdx: f, Threshold: thr}
	node.Left = BuildTree(leftX, leftY, params, depth+1, rng)
	node.Right = BuildTree(rightX, rightY, params, depth+1, rng)
	return node
}

// Predict recorre el árbol para clasificar una fila de features.
func Predict(node *Node, x []float64) int {
	for !node.IsLeaf {
		if x[node.FeatureIdx] <= node.Threshold {
			node = node.Left
		} else {
			node = node.Right
		}
	}
	return node.Prediction
}

// BootstrapSample genera una muestra con reemplazo del mismo
// tamaño que el dataset de entrenamiento (bagging clásico de
// Random Forest): cada árbol ve una versión distinta de los datos.
func BootstrapSample(X [][]float64, y []int, rng *rand.Rand) ([][]float64, []int) {
	n := len(y)
	bx := make([][]float64, n)
	by := make([]int, n)
	for i := 0; i < n; i++ {
		j := rng.Intn(n)
		bx[i] = X[j]
		by[i] = y[j]
	}
	return bx, by
}

// MajorityVote combina los votos (0/1) de todos los árboles del
// bosque para una fila, igual que hacía el Aggregator en Promela.
func MajorityVote(votes []int) int {
	sum := 0
	for _, v := range votes {
		sum += v
	}
	if sum*2 >= len(votes) {
		return 1
	}
	return 0
}

// Accuracy calcula el porcentaje de aciertos entre predicción y verdad.
func Accuracy(yTrue, yPred []int) float64 {
	if len(yTrue) == 0 {
		return 0
	}
	correct := 0
	for i := range yTrue {
		if yTrue[i] == yPred[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(yTrue))
}
