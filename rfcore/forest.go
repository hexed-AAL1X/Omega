package rfcore

import (
	"math/rand"
	"sync"
)

type Forest struct {
	Trees []*Node
}

func trainTree(X [][]float64, y []int, params TreeParams, seed int64, i int) *Node {
	rng := rand.New(rand.NewSource(seed + int64(i)))
	return BuildTree(X, y, balancedBootstrap(y, rng), params, rng)
}

func balancedBootstrap(y []int, rng *rand.Rand) []int {
	var pos, neg []int
	for i, v := range y {
		if v == 1 {
			pos = append(pos, i)
		} else {
			neg = append(neg, i)
		}
	}
	if len(pos) == 0 || len(neg) == 0 {
		idx := make([]int, len(y))
		for k := range idx {
			idx[k] = rng.Intn(len(y))
		}
		return idx
	}
	half := len(y) / 2
	idx := make([]int, 0, 2*half)
	for k := 0; k < half; k++ {
		idx = append(idx, pos[rng.Intn(len(pos))], neg[rng.Intn(len(neg))])
	}
	return idx
}

func TrainSequential(X [][]float64, y []int, nTrees int, params TreeParams, seed int64) *Forest {
	trees := make([]*Node, nTrees)
	for i := 0; i < nTrees; i++ {
		trees[i] = trainTree(X, y, params, seed, i)
	}
	return &Forest{Trees: trees}
}

func TrainConcurrent(X [][]float64, y []int, nTrees, workers int, params TreeParams, seed int64) *Forest {
	if workers < 1 {
		workers = 1
	}
	jobs := make(chan int, nTrees)
	for i := 0; i < nTrees; i++ {
		jobs <- i
	}
	close(jobs)

	var (
		mu    sync.Mutex
		wg    sync.WaitGroup
		trees = make([]*Node, nTrees)
		done  int
	)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				tree := trainTree(X, y, params, seed, i)
				mu.Lock()
				trees[i] = tree
				done++
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	if done != nTrees {
		panic("worker pool: no se entrenaron todos los árboles")
	}
	return &Forest{Trees: trees}
}

func (f *Forest) Predict(x []float64) int {
	ones := 0
	for _, t := range f.Trees {
		ones += Predict(t, x)
	}
	return majority(ones, len(f.Trees))
}

func (f *Forest) PredictAll(X [][]float64) []int {
	out := make([]int, len(X))
	for i, row := range X {
		out[i] = f.Predict(row)
	}
	return out
}
