package main

import (
	"math/rand"
)

// SGD support stochastic gradient decent through implementation of
// a random out-of-bag ChooseOne function
type SGD struct {
	samples []int `slice holds samples yet to be chosen`
	total   int
	pRNG    *rand.Rand
}

func NewSGD(numsamples int, pRNG *rand.Rand) *SGD {
	sgd := &SGD{
		total: numsamples,
		pRNG:  pRNG,
	}
	sgd.refreshSamples()
	return sgd
}

func (sgd *SGD) refreshSamples() {
	sgd.samples = make([]int, sgd.total)
	for idx := range sgd.samples {
		sgd.samples[idx] = idx
	}
}

// ChooseOne sample and remove from samples slice
// If the samples slice is spent, refresh
func (sgd *SGD) ChooseOne() (choice int, refresh bool) {
	numsamples := len(sgd.samples)
	idx := sgd.pRNG.Intn(numsamples)
	choice = sgd.samples[idx]
	if numsamples == 1 {
		sgd.refreshSamples()
		refresh = true
	} else {
		sgd.samples = append(sgd.samples[:idx], sgd.samples[idx+1:]...)
	}
	return choice, refresh
}

type Descender interface {
	ChooseOne() int
}
