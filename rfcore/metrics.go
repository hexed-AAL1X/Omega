package rfcore

import (
	"fmt"
	"math"
	"sort"
)

type Metrics struct {
	Accuracy         float64
	BalancedAccuracy float64
	RecallExito      float64
	RecallFracaso    float64
	PrecisionFracaso float64
	F1Fracaso        float64
	Baseline         float64
}

func Evaluate(yTrue, yPred []int) Metrics {
	var tp, tn, fp, fn int
	for i := range yTrue {
		switch {
		case yTrue[i] == 1 && yPred[i] == 1:
			tp++
		case yTrue[i] == 0 && yPred[i] == 0:
			tn++
		case yTrue[i] == 0 && yPred[i] == 1:
			fp++
		default:
			fn++
		}
	}
	n := float64(len(yTrue))
	m := Metrics{
		Accuracy:         float64(tp+tn) / n,
		RecallExito:      ratio(tp, tp+fn),
		RecallFracaso:    ratio(tn, tn+fp),
		PrecisionFracaso: ratio(tn, tn+fn),
		Baseline:         math.Max(float64(tp+fn), float64(tn+fp)) / n,
	}
	m.BalancedAccuracy = (m.RecallExito + m.RecallFracaso) / 2
	if m.PrecisionFracaso+m.RecallFracaso > 0 {
		m.F1Fracaso = 2 * m.PrecisionFracaso * m.RecallFracaso / (m.PrecisionFracaso + m.RecallFracaso)
	}
	return m
}

func ratio(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b)
}

func (m Metrics) String() string {
	return fmt.Sprintf(
		"exactitud=%.4f (base=%.4f) | exactitud balanceada=%.4f | recall éxito=%.4f | recall fracaso=%.4f | F1 fracaso=%.4f",
		m.Accuracy, m.Baseline, m.BalancedAccuracy, m.RecallExito, m.RecallFracaso, m.F1Fracaso,
	)
}

func TrimmedMean(vals []float64, trim float64) float64 {
	s := append([]float64(nil), vals...)
	sort.Float64s(s)
	k := int(float64(len(s)) * trim)
	s = s[k : len(s)-k]
	return Mean(s)
}

func Mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

func StdDev(vals []float64) float64 {
	if len(vals) < 2 {
		return 0
	}
	m := Mean(vals)
	sum := 0.0
	for _, v := range vals {
		sum += (v - m) * (v - m)
	}
	return math.Sqrt(sum / float64(len(vals)-1))
}

func MinMax(vals []float64) (float64, float64) {
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, v := range vals {
		lo, hi = math.Min(lo, v), math.Max(hi, v)
	}
	return lo, hi
}
