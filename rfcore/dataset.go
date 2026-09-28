package rfcore

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
)

var numericCols = []string{
	"start_year", "duration_years", "commitment_usd",
	"gdp_pc_ppp", "gdp_pc", "gdp_pc_growth", "elec_access", "child_mort", "unemp",
	"urban_pct", "internet_pct", "poverty", "literacy",
	"n_aid_projects", "n_donors", "usd_total_net", "usd_total_gross", "usd_ods17",
	"n_projects_ods17", "n_decommitments",
}

var categoricalCols = []string{"donor", "wb_region", "wb_income"}

var missingFlagCols = []string{"commitment_usd", "n_donors"}

type Dataset struct {
	X            [][]float64
	Y            []int
	FeatureNames []string
}

func LoadCSV(path string, soloVentana bool) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("no se pudo abrir %s: %w", path, err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	header, err := r.Read()
	if err != nil {
		return nil, fmt.Errorf("no se pudo leer la cabecera: %w", err)
	}
	col := make(map[string]int, len(header))
	for i, h := range header {
		col[h] = i
	}
	required := append(append([]string{"fila_ok", "in_financing_window", "success"}, numericCols...), categoricalCols...)
	for _, name := range required {
		if _, ok := col[name]; !ok {
			return nil, fmt.Errorf("falta la columna %q en %s", name, path)
		}
	}

	var records [][]string
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("error leyendo %s: %w", path, err)
		}
		if rec[col["fila_ok"]] != "true" {
			continue
		}
		if soloVentana && rec[col["in_financing_window"]] != "true" {
			continue
		}
		records = append(records, rec)
	}
	if len(records) == 0 {
		return nil, fmt.Errorf("no quedaron filas tras filtrar fila_ok")
	}

	categories := make(map[string][]string, len(categoricalCols))
	for _, c := range categoricalCols {
		seen := map[string]bool{}
		for _, rec := range records {
			if v := rec[col[c]]; v != "" {
				seen[v] = true
			}
		}
		for v := range seen {
			categories[c] = append(categories[c], v)
		}
		sort.Strings(categories[c])
	}

	names := append([]string(nil), numericCols...)
	for _, c := range missingFlagCols {
		names = append(names, "falta_"+c)
	}
	for _, c := range categoricalCols {
		for _, v := range categories[c] {
			names = append(names, c+"="+v)
		}
	}

	ds := &Dataset{
		X:            make([][]float64, 0, len(records)),
		Y:            make([]int, 0, len(records)),
		FeatureNames: names,
	}
	for _, rec := range records {
		row := make([]float64, 0, len(names))
		for _, c := range numericCols {
			row = append(row, parseFloat(rec[col[c]]))
		}
		for _, c := range missingFlagCols {
			if rec[col[c]] == "" {
				row = append(row, 1)
			} else {
				row = append(row, 0)
			}
		}
		for _, c := range categoricalCols {
			v := rec[col[c]]
			for _, cat := range categories[c] {
				if v == cat {
					row = append(row, 1)
				} else {
					row = append(row, 0)
				}
			}
		}
		label, err := strconv.Atoi(rec[col["success"]])
		if err != nil || (label != 0 && label != 1) {
			return nil, fmt.Errorf("success inválido: %q", rec[col["success"]])
		}
		ds.X = append(ds.X, row)
		ds.Y = append(ds.Y, label)
	}
	return ds, nil
}

func parseFloat(s string) float64 {
	if s == "" {
		return math.NaN()
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return math.NaN()
	}
	return v
}

type Split struct {
	TrainX, TestX [][]float64
	TrainY, TestY []int
}

func TrainTestSplit(ds *Dataset, testFrac float64, seed int64) Split {
	idx := rand.New(rand.NewSource(seed)).Perm(len(ds.X))
	nTest := int(float64(len(idx)) * testFrac)

	var s Split
	for k, i := range idx {
		row := append([]float64(nil), ds.X[i]...)
		if k < nTest {
			s.TestX = append(s.TestX, row)
			s.TestY = append(s.TestY, ds.Y[i])
		} else {
			s.TrainX = append(s.TrainX, row)
			s.TrainY = append(s.TrainY, ds.Y[i])
		}
	}
	imputeMedian(s.TrainX, s.TestX)
	return s
}

func imputeMedian(train, test [][]float64) {
	if len(train) == 0 {
		return
	}
	for j := range train[0] {
		vals := make([]float64, 0, len(train))
		for _, row := range train {
			if !math.IsNaN(row[j]) {
				vals = append(vals, row[j])
			}
		}
		med := 0.0
		if len(vals) > 0 {
			sort.Float64s(vals)
			med = vals[len(vals)/2]
		}
		for _, rows := range [][][]float64{train, test} {
			for _, row := range rows {
				if math.IsNaN(row[j]) {
					row[j] = med
				}
			}
		}
	}
}
