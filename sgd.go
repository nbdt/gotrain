package main

import (
	"math/rand"
)

// SGD support stochastic gradient decent through implementation of
// a random out-of-bag ChooseOne function
type SGD struct {
	samples []int `slice holds samples yet to be chosen`
	total   int
	refresh bool
	pRNG    *rand.Rand
}

func NewSGD(numsamples int, pRNG *rand.Rand) Descender {
	sgd := &SGD{
		total:   numsamples,
		pRNG:    pRNG,
		refresh: false,
	}
	sgd.refreshSamples()
	sgd.refresh = false
	return sgd
}

func (sgd *SGD) refreshSamples() {
	sgd.refresh = true
	sgd.samples = make([]int, sgd.total)
	for idx := range sgd.samples {
		sgd.samples[idx] = idx
	}
}

// ChooseOne sample and remove from samples slice
// If the samples slice is spent, refresh
func (sgd *SGD) ChooseOne() int {
	numsamples := len(sgd.samples)
	idx := sgd.pRNG.Intn(numsamples)
	choice := sgd.samples[idx]
	switch numsamples {
	case 1:
		sgd.refreshSamples()
	default:
		sgd.samples = append(sgd.samples[:idx], sgd.samples[idx+1:]...)
	}
	return choice
}

func (sgd *SGD) ChooseMany(numBatches int) (batches [][]int) {
	batches = make([][]int, numBatches)
	for i := len(sgd.samples); i > 0; i-- {
		batches[i%numBatches] = append(batches[i%numBatches], sgd.ChooseOne())
	}
	sgd.refreshSamples()
	return batches
}

func (sgd *SGD) SampledAll() bool {
	switch sgd.refresh {
	case false:
		return false
	default:
		sgd.refresh = false
		return true
	}
}

type Descender interface {
	ChooseOne() int
	ChooseMany(numBatches int) [][]int
	SampledAll() bool
}
