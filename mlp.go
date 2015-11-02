package main

import (
	"fmt"
	"math"
	"os"
	"time"
)

var modelMLP = &Model{
	Train:       trainMLPSyncronousMiniBatch,
	Description: `MLP -MultiLayer Perceptrontrain feed-forward neural network with back propagation`,
}

var (
	epsillon     = 1.0 / float64(1<<23)
	momentum     float64
	learningRate float64
	costThresh   float64
	maxEpochs    int
	numnets      int
	batchsize    int
)

func init() {
	modelMLP.Flag.Float64Var(&momentum, "momentum", 0.0, "Momentum of network (default 0)")
	modelMLP.Flag.Float64Var(&learningRate, "learningRate", 1.0, "Learning rate of the of network")
	modelMLP.Flag.Float64Var(&costThresh, "costThresh", -math.MaxFloat64, "Minimum epoch error")
	modelMLP.Flag.IntVar(&maxEpochs, "maxEpochs", math.MaxInt64, "Maximum epoch training cycles")
	modelMLP.Flag.IntVar(&batchsize, "batchsize", 1, "number of batches within an epoch")
	if momentum < 0.0 || momentum > 1.0 {
		fmt.Fprintf(os.Stderr, "momentum [%v] is not between 0.0 and 1.0\n", momentum)
		os.Exit(2)
	}
	if learningRate < 0.0 || learningRate > 1.0 {
		fmt.Fprintf(os.Stderr, "learning rate [%v] is not between 0.0 and 1.0\n", learningRate)
		os.Exit(2)
	}
}

// NewMLP constructs the convolutional neuro network based on the topology described in
// Add* and Set* options
func NewMLP(insize, outsize int) *Network {
	// Define weight initialization function
	n := NewNetwork(SetLearningRate(learningRate),
		SetMomentum(momentum),
		SetCostor(CrossEntropy),
		SetWeightInitFunc(Normal(0.0, 0.01)),
		AddInputLayer(insize, true),
		AddLayer(50, true, Rectlin),
		AddOutputLayer(outsize, Sigmoid))
	return n
}

// fMLP runs the feed forward pass based on inputs and produces outputs of the network
func fMLP(n *Network, inputs []float64) (outputs []float64) {
	// Set input layer values
	for idx, val := range inputs {
		n.Layers[0].Neurons[idx].Value = val
	}
	for idx, layer := range n.Layers {
		for _, neuron := range layer.Neurons {
			for _, cxn := range neuron.Connections {
				cxn.toNeuron.Value += neuron.Value * cxn.Weight
			}
		}
		if layer.Bias != nil {
			for _, cxn := range layer.Bias.Connections {
				cxn.toNeuron.Value += layer.Bias.OUT * cxn.Weight
			}
		}
		// Apply the activation function to the forward layer
		if idx < n.numlayers-1 {
			for _, neuron := range n.Layers[idx+1].Neurons {
				neuron.Value = n.Layers[idx+1].A.F(neuron.Value)
			}
		}
	}
	// Get values from output neurons to return
	outputs = make([]float64, n.numout)
	for idx, outputneuron := range n.Layers[n.numlayers-1].Neurons {
		outputs[idx] = outputneuron.Value
	}
	return outputs
}

// backPropagate adjusts the connection weights of the network based on the error, Delta,
// of each neuron computed during gradient decent
func bMLP(n *Network, targets []float64) {
	var (
		activatePrime float64
		costPrime     float64
		delta         float64
	)
	outlayer := n.Layers[n.numlayers-1]
	// Calculate delta of output layer
	if outlayer.shortcut {
		for idx, neuron := range outlayer.Neurons {
			costPrime = n.C.FPrime(neuron.Value, targets[idx])
			neuron.Delta = costPrime
		}
	} else {
		for idx, neuron := range outlayer.Neurons {
			costPrime = n.C.FPrime(neuron.Value, targets[idx])
			activatePrime = outlayer.A.FPrime(neuron.Value)
			neuron.Delta = costPrime * activatePrime
		}
	}
	// Calculate delta of hidden layers
	for idx := n.numlayers - 2; idx >= 1; idx-- {
		layer := n.Layers[idx]
		for _, neuron := range layer.Neurons {
			for _, cxn := range neuron.Connections {
				delta = cxn.Weight * cxn.toNeuron.Delta
				neuron.Delta += delta
			}
			delta = layer.A.FPrime(neuron.Value)
			neuron.Delta *= delta
		}
		if layer.Bias != nil {
			for _, cxn := range layer.Bias.Connections {
				delta = cxn.Weight * cxn.toNeuron.Delta
				layer.Bias.Delta += delta
			}
			delta = layer.A.FPrime(layer.Bias.OUT)
			layer.Bias.Delta *= delta
		}
	}
}

func accumulateWeightChange(n *Network) {
	var weightChange float64
	for _, layer := range n.Layers[0 : n.numlayers-1] {
		for _, neuron := range layer.Neurons {
			for _, cxn := range neuron.Connections {
				weightChange = n.eta * neuron.Value * cxn.toNeuron.Delta
				cxn.weightChange += weightChange
			}
		}
		if layer.Bias != nil {
			for _, cxn := range layer.Bias.Connections {
				weightChange = n.eta * layer.Bias.OUT * cxn.toNeuron.Delta
				cxn.weightChange += weightChange
			}
		}
	}
}

func updateWeights(n *Network) {
	var (
		weightChange float64
		momentumTerm float64
		bze          = float64(batchsize)
	)
	// Apply deltas to connection weights
	for _, layer := range n.Layers {
		for _, neuron := range layer.Neurons {
			for _, cxn := range neuron.Connections {
				weightChange = cxn.weightChange / bze
				momentumTerm = n.momentum * cxn.prevChange
				weightChange += momentumTerm
				cxn.Weight -= weightChange
				cxn.prevChange = weightChange
				cxn.weightChange = 0.0
			}
		}
		if layer.Bias != nil {
			for _, cxn := range layer.Bias.Connections {
				weightChange = cxn.weightChange / bze
				momentumTerm = n.momentum * cxn.prevChange
				weightChange += momentumTerm
				cxn.Weight -= weightChange
				cxn.prevChange = weightChange
				cxn.weightChange = 0.0
			}
		}
	}
}

type MLPExec struct {
	network *Network
}

// execMLP prediction on input given a trained network
func (mlp *MLPExec) Execute(input []float64) []float64 {
	output := fMLP(mlp.network, input)
	mlp.network.zeroValues()
	return output
}

func trainMLPSyncronousMiniBatch(d *TrainData) (e Executor) {
	mlp := NewMLP(d.FeatureLen, d.OutputLen)
	ts := NewTrainStatus()
	sgd := NewSGD(d.SampleLen, pRNG)
	numbatches := d.SampleLen / batchsize
	fmt.Fprintf(os.Stderr, "Beginning training with:\n%v", mlp)
	fmt.Fprintf(os.Stderr, "MaxEpochs = %v and CostThreshold = %v\n", maxEpochs, costThresh)
	fmt.Fprintf(os.Stderr, "BatchCount = %v\n\n", numbatches)
	for ts.epoch < maxEpochs && ts.epochCost > costThresh && !isStopEarly() {
		ts.epochTime = time.Now()
		batches := sgd.ChooseMany(numbatches)
		for _, batch := range batches {
			for _, sampleIDX := range batch {
				output := fMLP(mlp, d.Input[sampleIDX])
				expected := d.Output[sampleIDX]
				for idx, val := range output {
					ts.sampCost += mlp.C.F(val, expected[idx]) / mlp.costDivisor
				}
				ts.epochCost += (ts.sampCost / d.SampleLength)
				ts.sampCost = 0.0
				bMLP(mlp, expected)
				accumulateWeightChange(mlp)
				mlp.zeroValuesAndDeltas()
			}
			updateWeights(mlp)
		}
		fmt.Fprintf(os.Stderr, "Epoch took %v\n", time.Now().Sub(ts.epochTime))
		fmt.Fprintf(os.Stderr, "Epoch: %v, Cost: %.4f\n", ts.epoch, ts.epochCost)
		fmt.Fprintf(os.Stderr, "%v\n", mlp)
		ts.epoch++
		ts.finalCost = ts.epochCost
		ts.epochCost, ts.sampCost = 0.0, 0.0
	}
	fmt.Fprintf(os.Stderr, "Finished training of %v\n", mlp)
	fmt.Fprintf(os.Stderr, "Epoch: %v, Cost: %.4f\n", ts.epoch, ts.finalCost)
	return &MLPExec{
		network: mlp,
	}
}
