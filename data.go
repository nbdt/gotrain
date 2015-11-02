package main

import (
	"encoding/csv"
	"math"
	"os"
	"strconv"
	"strings"
)

type TrainData struct {
	Input        [][]float64
	FeatureLen   int
	SampleLength float64
	SampleLen    int
	Output       [][]float64
	OutputLength float64
	OutputLen    int
}

func (td *TrainData) HasInputs() bool {
	return len(td.Input) > 0
}

func (td *TrainData) HasOutputs() bool {
	return len(td.Output) > 0
}

func (td *TrainData) InputIsNormal() bool {
	if !td.HasInputs() {
		return false
	}
	return td.isNormal(td.Input)
}

func (td *TrainData) OutputIsNormal() bool {
	if !td.HasOutputs() {
		return false
	}
	return td.isNormal(td.Output)
}

func (td *TrainData) isNormal(X [][]float64) bool {
	zeroSize := len(X[0])
	for _, x := range X {
		if len(x) != zeroSize {
			return false
		}
	}
	return true
}

// ReadCSV reads in data from csv file
// assuming that all classes come first in the csv
// Ex: c0, ..., cn, f0, ..., fn
func ReadCSV(datapath string, numclasses int) (*TrainData, error) {
	file, err := os.Open(datapath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)

	d := &TrainData{}
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	for _, row := range rows {
		in := row[numclasses:]
		inf := make([]float64, len(in))
		for idx := range inf {
			inf[idx], err = strconv.ParseFloat(strings.TrimSpace(in[idx]), 64)
			if err != nil {
				return nil, err
			}
		}
		d.Input = append(d.Input, Normalize(inf))

		out := row[:numclasses]
		outf := make([]float64, len(out))
		for idx := range outf {
			outf[idx], err = strconv.ParseFloat(strings.TrimSpace(out[idx]), 64)
			if err != nil {
				return nil, err
			}
		}
		d.Output = append(d.Output, outf)
	}
	d.FeatureLen = len(d.Input[0])
	d.SampleLength = float64(len(d.Input))
	d.SampleLen = len(d.Input)
	d.OutputLength = float64(len(d.Output[0]))
	d.OutputLen = len(d.Output[0])
	return d, nil
}

func Normalize(d []float64) []float64 {
	var (
		mean float64
		std  float64
	)
	numsamp := len(d)
	for _, v := range d {
		mean += v
	}
	mean /= float64(numsamp)
	for _, v := range d {
		std += math.Pow(v-mean, 2)
	}
	std /= float64(numsamp)
	std = math.Sqrt(std)
	res := make([]float64, numsamp)
	for idx, v := range d {
		//res[idx] = (v - mean) / std
		res[idx] = v / 255.0
	}
	return res
}

func isStopEarly() bool {
	_, err := os.Stat("./stopearly")
	return !os.IsNotExist(err)
}
