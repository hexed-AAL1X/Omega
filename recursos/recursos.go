package recursos

import (
	"runtime"
	"runtime/metrics"
	"sync"
	"time"
)

type Medicion struct {
	Segundos    float64
	CPUSegundos float64
	UsoCPU      float64
	AllocMB     float64
	HeapPicoMB  float64
}

var nombres = []string{
	"/cpu/classes/total:cpu-seconds",
	"/cpu/classes/idle:cpu-seconds",
	"/gc/heap/allocs:bytes",
	"/memory/classes/heap/objects:bytes",
}

func leer() (cpu, alloc, heap float64) {
	s := make([]metrics.Sample, len(nombres))
	for i, n := range nombres {
		s[i].Name = n
	}
	metrics.Read(s)
	valor := func(m metrics.Sample) float64 {
		switch m.Value.Kind() {
		case metrics.KindFloat64:
			return m.Value.Float64()
		case metrics.KindUint64:
			return float64(m.Value.Uint64())
		}
		return 0
	}
	return valor(s[0]) - valor(s[1]), valor(s[2]), valor(s[3])
}

func Medir(f func()) Medicion {
	runtime.GC()
	cpu0, alloc0, _ := leer()

	var (
		mu   sync.Mutex
		pico float64
		fin  = make(chan struct{})
		wg   sync.WaitGroup
	)
	wg.Add(1)
	go func() {
		defer wg.Done()
		t := time.NewTicker(10 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-fin:
				return
			case <-t.C:
				_, _, h := leer()
				mu.Lock()
				if h > pico {
					pico = h
				}
				mu.Unlock()
			}
		}
	}()

	inicio := time.Now()
	f()
	seg := time.Since(inicio).Seconds()
	close(fin)
	wg.Wait()

	_, _, h := leer()
	runtime.GC()
	cpu1, alloc1, _ := leer()
	if h > pico {
		pico = h
	}
	cpuSeg := cpu1 - cpu0
	return Medicion{
		Segundos:    seg,
		CPUSegundos: cpuSeg,
		UsoCPU:      cpuSeg / (seg * float64(runtime.NumCPU())),
		AllocMB:     (alloc1 - alloc0) / (1 << 20),
		HeapPicoMB:  pico / (1 << 20),
	}
}
