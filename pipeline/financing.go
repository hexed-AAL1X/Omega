package pipeline

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"strconv"
	"sync"
)

const nGoals = 17

type Clave struct {
	Receptor string
	Anio     int
}

type Agregado struct {
	Filas        int
	Donantes     map[string]struct{}
	USDNeto      float64
	USDBruto     float64
	Descompromis int
	Goals        [nGoals]float64
}

type Resultado struct {
	Filas     int
	Invalidas int
	Grupos    map[Clave]*Agregado
}

type columnas struct {
	anio, donante, receptor, monto int
	goals                          [nGoals]int
}

func leerCabecera(data []byte) (columnas, []byte, error) {
	fin := bytes.IndexByte(data, '\n')
	if fin < 0 {
		return columnas{}, nil, fmt.Errorf("archivo sin cabecera")
	}
	header, err := csv.NewReader(bytes.NewReader(data[:fin+1])).Read()
	if err != nil {
		return columnas{}, nil, err
	}
	idx := map[string]int{}
	for i, h := range header {
		idx[h] = i
	}
	buscar := func(n string) (int, error) {
		i, ok := idx[n]
		if !ok {
			return 0, fmt.Errorf("falta la columna %q", n)
		}
		return i, nil
	}
	var c columnas
	for _, par := range []struct {
		dst  *int
		name string
	}{{&c.anio, "year"}, {&c.donante, "donor"}, {&c.receptor, "recipient"}, {&c.monto, "commitment_amount_usd_constant"}} {
		if *par.dst, err = buscar(par.name); err != nil {
			return columnas{}, nil, err
		}
	}
	for g := 0; g < nGoals; g++ {
		if c.goals[g], err = buscar(fmt.Sprintf("goal_%d", g+1)); err != nil {
			return columnas{}, nil, err
		}
	}
	return c, data[fin+1:], nil
}

func numero(s string) float64 {
	if s == "" || s == "NA" {
		return 0
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsNaN(v) {
		return 0
	}
	return v
}

func procesarBloque(bloque []byte, c columnas) (Resultado, error) {
	res := Resultado{Grupos: map[Clave]*Agregado{}}
	r := csv.NewReader(bytes.NewReader(bloque))
	r.ReuseRecord = true
	for {
		rec, err := r.Read()
		if err == io.EOF {
			return res, nil
		}
		if err != nil {
			return res, err
		}
		res.Filas++
		anio, err := strconv.Atoi(rec[c.anio])
		if err != nil || rec[c.receptor] == "" {
			res.Invalidas++
			continue
		}
		k := Clave{Receptor: rec[c.receptor], Anio: anio}
		a := res.Grupos[k]
		if a == nil {
			a = &Agregado{Donantes: map[string]struct{}{}}
			res.Grupos[k] = a
		}
		a.Filas++
		if _, ok := a.Donantes[rec[c.donante]]; !ok {
			a.Donantes[rec[c.donante]] = struct{}{}
		}
		monto := numero(rec[c.monto])
		a.USDNeto += monto
		if monto > 0 {
			a.USDBruto += monto
		}
		if monto < 0 {
			a.Descompromis++
		}
		for g := 0; g < nGoals; g++ {
			a.Goals[g] += numero(rec[c.goals[g]])
		}
	}
}

func combinar(dst *Resultado, src Resultado) {
	dst.Filas += src.Filas
	dst.Invalidas += src.Invalidas
	for k, s := range src.Grupos {
		d := dst.Grupos[k]
		if d == nil {
			dst.Grupos[k] = s
			continue
		}
		d.Filas += s.Filas
		for don := range s.Donantes {
			d.Donantes[don] = struct{}{}
		}
		d.USDNeto += s.USDNeto
		d.USDBruto += s.USDBruto
		d.Descompromis += s.Descompromis
		for g := 0; g < nGoals; g++ {
			d.Goals[g] += s.Goals[g]
		}
	}
}

func Secuencial(data []byte) (Resultado, error) {
	c, cuerpo, err := leerCabecera(data)
	if err != nil {
		return Resultado{}, err
	}
	return procesarBloque(cuerpo, c)
}

func partir(cuerpo []byte, n int) [][]byte {
	if n < 1 {
		n = 1
	}
	var bloques [][]byte
	tam := len(cuerpo) / n
	inicio := 0
	for i := 0; i < n && inicio < len(cuerpo); i++ {
		fin := inicio + tam
		if i == n-1 || fin >= len(cuerpo) {
			fin = len(cuerpo)
		} else if j := bytes.IndexByte(cuerpo[fin:], '\n'); j >= 0 {
			fin += j + 1
		} else {
			fin = len(cuerpo)
		}
		bloques = append(bloques, cuerpo[inicio:fin])
		inicio = fin
	}
	return bloques
}

func Concurrente(data []byte, workers, bloques int) (Resultado, error) {
	c, cuerpo, err := leerCabecera(data)
	if err != nil {
		return Resultado{}, err
	}
	if workers < 1 {
		workers = 1
	}

	partes := partir(cuerpo, bloques)
	tareas := make(chan []byte, len(partes))
	for _, p := range partes {
		tareas <- p
	}
	close(tareas)

	parciales := make(chan Resultado, workers)
	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		primerE error
	)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			local := Resultado{Grupos: map[Clave]*Agregado{}}
			for bloque := range tareas {
				r, err := procesarBloque(bloque, c)
				if err != nil {
					mu.Lock()
					if primerE == nil {
						primerE = err
					}
					mu.Unlock()
					continue
				}
				combinar(&local, r)
			}
			parciales <- local
		}()
	}
	go func() {
		wg.Wait()
		close(parciales)
	}()

	total := Resultado{Grupos: map[Clave]*Agregado{}}
	for p := range parciales {
		combinar(&total, p)
	}
	return total, primerE
}

func Recortar(data []byte, frac float64) []byte {
	if frac >= 1 {
		return data
	}
	fin := int(float64(len(data)) * frac)
	if j := bytes.IndexByte(data[fin:], '\n'); j >= 0 {
		return data[:fin+j+1]
	}
	return data
}

func Iguales(a, b Resultado) error {
	if a.Filas != b.Filas || a.Invalidas != b.Invalidas || len(a.Grupos) != len(b.Grupos) {
		return fmt.Errorf("filas %d/%d, inválidas %d/%d, grupos %d/%d",
			a.Filas, b.Filas, a.Invalidas, b.Invalidas, len(a.Grupos), len(b.Grupos))
	}
	cerca := func(x, y float64) bool {
		return math.Abs(x-y) <= 1e-6*math.Max(1, math.Max(math.Abs(x), math.Abs(y)))
	}
	for k, x := range a.Grupos {
		y, ok := b.Grupos[k]
		if !ok || x.Filas != y.Filas || len(x.Donantes) != len(y.Donantes) || x.Descompromis != y.Descompromis ||
			!cerca(x.USDNeto, y.USDNeto) || !cerca(x.USDBruto, y.USDBruto) {
			return fmt.Errorf("el grupo %v no coincide", k)
		}
		for g := 0; g < nGoals; g++ {
			if !cerca(x.Goals[g], y.Goals[g]) {
				return fmt.Errorf("goal_%d del grupo %v no coincide", g+1, k)
			}
		}
	}
	return nil
}
