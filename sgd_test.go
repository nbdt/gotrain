package main

import (
	"math/rand"
	"sort"
	"testing"
)

func TestSGD_ChooseManyEven(t *testing.T) {
	batchsize = 128
	samplesize := 32000
	source := rand.NewSource(1337)
	pRNG := rand.New(source)
	sgd := NewSGD(samplesize, pRNG)
	batches := sgd.ChooseMany(samplesize / batchsize)
	if len(batches) != samplesize/batchsize {
		t.Errorf("Expected %v Got %v For batchsize = %v and samplesize = %v",
			samplesize/batchsize, len(batches), batchsize, samplesize)
	}
	var samples sort.IntSlice
	for _, batch := range batches {
		if len(batch) < batchsize-1 || len(batch) > batchsize+1 {
			t.Errorf("Expected %v or %v Got %v For numbatches = %v and samplesize = %v",
				batchsize, batchsize-1, len(batch), samplesize/batchsize, samplesize)
		}
		for _, sample := range batch {
			samples = append(samples, sample)
		}
	}
	samples.Sort()
	for idx := range samples {
		if idx != samples[idx] {
			t.Error("Missing sample %v in batches", idx)
		}
	}
}

func TestSGD_ChooseManyOdd(t *testing.T) {
	batchsize = 10
	samplesize := 105
	source := rand.NewSource(1337)
	pRNG := rand.New(source)
	sgd := NewSGD(samplesize, pRNG)
	batches := sgd.ChooseMany(samplesize / batchsize)
	if len(batches) != samplesize/batchsize {
		t.Errorf("Expected %v Got %v For batchsize = %v and samplesize = %v",
			samplesize/batchsize, len(batches), batchsize, samplesize)
	}
	var samples sort.IntSlice
	for _, batch := range batches {
		if len(batch) < batchsize-1 || len(batch) > batchsize+1 {
			t.Errorf("Expected %v or %v Got %v For numbatches = %v and samplesize = %v",
				batchsize, batchsize-1, len(batch), samplesize/batchsize, samplesize)
		}
		for _, sample := range batch {
			samples = append(samples, sample)
		}
	}
	samples.Sort()
	for idx := range samples {
		if idx != samples[idx] {
			t.Error("Missing sample %v in batches", idx)
		}
	}
}
