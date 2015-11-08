package main

import (
	"encoding/csv"
	"math"
	"os"
	"strconv"
	"strings"
)

type TrainData struct {
	Input        [][]float32
	FeatureLen   int
	SampleLength float32
	SampleLen    int
	Output       [][]float32
	OutputLength float32
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

func (td *TrainData) isNormal(X [][]float32) bool {
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
		inf := make([]float32, len(in))
		for idx := range inf {
			val, err := strconv.ParseFloat(strings.TrimSpace(in[idx]), 32)
			if err != nil {
				return nil, err
			}
			inf[idx] = float32(val)
		}
		d.Input = append(d.Input, Normalize(inf))

		out := row[:numclasses]
		outf := make([]float32, len(out))
		for idx := range outf {
			val, err := strconv.ParseFloat(strings.TrimSpace(out[idx]), 32)
			if err != nil {
				return nil, err
			}
			outf[idx] = float32(val)
		}
		d.Output = append(d.Output, outf)
	}
	d.FeatureLen = len(d.Input[0])
	d.SampleLength = float32(len(d.Input))
	d.SampleLen = len(d.Input)
	d.OutputLength = float32(len(d.Output[0]))
	d.OutputLen = len(d.Output[0])
	return d, nil
}

func Normalize(d []float32) []float32 {
	var (
		mean float32
		std  float32
	)
	numsamp := len(d)
	for _, v := range d {
		mean += v
	}
	mean /= float32(numsamp)
	for _, v := range d {
		std += float32(math.Pow(float64(v-mean), 2))
	}
	std /= float32(numsamp)
	std = float32(math.Sqrt(float64(std)))
	res := make([]float32, numsamp)
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
