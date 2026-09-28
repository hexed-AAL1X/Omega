package rfcore

import (
	"math"
	"math/rand"
	"sort"
)

type TreeParams struct {
	MaxDepth        int
	MinSamplesSplit int
	MinSamplesLeaf  int
	MaxFeatures     int
}

func DefaultTreeParams(nFeatures int) TreeParams {
	mf := int(math.Sqrt(float64(nFeatures)))
	if mf < 1 {
		mf = 1
	}
	return TreeParams{MaxDepth: 10, MinSamplesSplit: 20, MinSamplesLeaf: 5, MaxFeatures: mf}
}

type Node struct {
	IsLeaf     bool
	Prediction int
	FeatureIdx int
	Threshold  float64
	Left       *Node
	Right      *Node
}

func BuildTree(X [][]float64, y []int, idx []int, params TreeParams, rng *rand.Rand) *Node {
	b := builder{X: X, y: y, params: params, rng: rng}
	return b.build(idx, 0)
}

type builder struct {
	X      [][]float64
	y      []int
	params TreeParams
	rng    *rand.Rand
}

func (b *builder) build(idx []int, depth int) *Node {
	ones := 0
	for _, i := range idx {
		ones += b.y[i]
	}
	n := len(idx)
	leaf := &Node{IsLeaf: true, Prediction: majority(ones, n)}
	if depth >= b.params.MaxDepth || n < b.params.MinSamplesSplit || ones == 0 || ones == n {
		return leaf
	}

	feat, thr, ok := b.bestSplit(idx, ones)
	if !ok {
		return leaf
	}

	left := make([]int, 0, n)
	right := make([]int, 0, n)
	for _, i := range idx {
		if b.X[i][feat] <= thr {
			left = append(left, i)
		} else {
			right = append(right, i)
		}
	}
	return &Node{
		FeatureIdx: feat,
		Threshold:  thr,
		Left:       b.build(left, depth+1),
		Right:      b.build(right, depth+1),
	}
}

func (b *builder) bestSplit(idx []int, totalOnes int) (int, float64, bool) {
	n := len(idx)
	nFeatures := len(b.X[0])
	k := b.params.MaxFeatures
	if k > nFeatures {
		k = nFeatures
	}
	candidates := b.rng.Perm(nFeatures)[:k]

	bestGini := gini(totalOnes, n)
	bestFeat, bestThr, found := 0, 0.0, false
	sorted := make([]int, n)
	minLeaf := b.params.MinSamplesLeaf

	for _, f := range candidates {
		copy(sorted, idx)
		sort.Slice(sorted, func(a, c int) bool { return b.X[sorted[a]][f] < b.X[sorted[c]][f] })

		leftOnes := 0
		for pos := 0; pos < n-1; pos++ {
			leftOnes += b.y[sorted[pos]]
			nLeft := pos + 1
			v, next := b.X[sorted[pos]][f], b.X[sorted[pos+1]][f]
			if v == next || nLeft < minLeaf || n-nLeft < minLeaf {
				continue
			}
			nRight := n - nLeft
			g := (float64(nLeft)*gini(leftOnes, nLeft) + float64(nRight)*gini(totalOnes-leftOnes, nRight)) / float64(n)
			if g < bestGini {
				bestGini, bestFeat, bestThr, found = g, f, (v+next)/2, true
			}
		}
	}
	return bestFeat, bestThr, found
}

func gini(ones, n int) float64 {
	if n == 0 {
		return 0
	}
	p := float64(ones) / float64(n)
	return 2 * p * (1 - p)
}

func majority(ones, n int) int {
	if 2*ones >= n {
		return 1
	}
	return 0
}

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
