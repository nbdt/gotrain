package main

import (
	"fmt"
	"os"
)

func doValidation(m Executor) {
	right := 0
	wrong := 0
	validationData, err := ReadCSV(*validatedatapath, *outputs)
	if err != nil {
		fmt.Printf("%s\n", err)
		os.Exit(1)
	}
	for idx, input := range validationData.Input {
		prediction := maxValIDX(m.Execute(input))
		actual := maxValIDX(validationData.Output[idx])
		if prediction != actual {
			wrong++
		} else {
			right++
		}
	}
	fmt.Printf("%.4f%% out of sample accuracy\n", 100.0*float64(right)/float64(right+wrong))

	right = 0
	wrong = 0
	sampleData, err := ReadCSV(*traindatapath, *outputs)
	if err != nil {
		fmt.Printf("%s\n", err)
		os.Exit(1)
	}
	for idx, input := range sampleData.Input {
		prediction := maxValIDX(m.Execute(input))
		actual := maxValIDX(sampleData.Output[idx])
		if prediction != actual {
			wrong++
		} else {
			right++
		}
	}
	fmt.Printf("%.4f%% in sample accuracy\n", 100.0*float64(right)/float64(right+wrong))
}

func maxValIDX(in []float64) int {
	maxval := 0.0
	maxidx := 0
	for idx, val := range in {
		if val > maxval {
			maxval = val
			maxidx = idx
		}
	}
	return maxidx
}
