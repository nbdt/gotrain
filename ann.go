package main

import (
	"fmt"
	"math"
)

// Connection to forward layer
type Connection struct {
	Weight     float64
	toNeuron   *Neuron
	prevChange float64
}

// Neuron in any layer
type Neuron struct {
	Value       float64
	Delta       float64
	Connections []*Connection
}

// Bias is a neuron with a constant value of 1.0
// and a constant output of AF(1.0).
// This provides the constant term in the linear
// equation of layers where bias is included
type Bias struct {
	ONE         float64
	OUT         float64
	Delta       float64
	Connections []*Connection
}

// Layer holds Neurons and Bias as well as functions
// for feedforward and back propigation step
type Layer struct {
	Neurons    []*Neuron
	Bias       *Bias
	A          Activator
	numneurons int
}

// AddLayer to the network with n neurons, possibly a bias, an activation function
// and the dirivative of the activation function as a function of the activation function.
// Pass this function into NewNetwork to add a layer to the network.
func AddLayer(numneurons int, includeBias bool, a Activator) func(*Network) {
	var bias *Bias
	if includeBias {
		bias = &Bias{ONE: 1.0,
			OUT: a.F(1.0),
		}
	}
	neurons := make([]*Neuron, numneurons)
	for idx := range neurons {
		neurons[idx] = &Neuron{}
	}
	layer := &Layer{
		Neurons:    neurons,
		Bias:       bias,
		A:          a,
		numneurons: numneurons,
	}
	return func(n *Network) {
		n.Layers = append(n.Layers, layer)
		n.numlayers++
	}
}

// SettErrorFunction of the network.
// Pass this function into NewNetwork to set the error function of the network
func SetErrorer(e ErrFuncer) func(*Network) {
	return func(n *Network) {
		n.EF = e.F
		n.EFPrime = e.FPrime
	}
}

// SetLearningRate of the network. Range 0.0 to 1.0, default is 1.0
// Pass this function into NewNetwork to set the learning rate.
func SetLearningRate(eta float64) func(*Network) {
	return func(n *Network) {
		n.eta = eta
	}
}

// SetMomentum of the network. Range 0.0 to 1.0, default is 0.0
// Pass this function into NewNetwork to set the Momentum.
func SetMomentum(momentum float64) func(*Network) {
	return func(n *Network) {
		n.momentum = momentum
	}
}

// SetWeightInitFunc of the network.
// Pass this function into NewNetwork to set the weight initalization function.
func SetWeightInitFunc(WeightInitFunc func() float64) func(*Network) {
	return func(n *Network) {
		n.WeightInitFunc = WeightInitFunc
	}
}

type TrainStatus struct {
	epochErr float64
	epoch    int
}

func (ts *TrainStatus) String() string {
	return fmt.Sprintf("Epoch %v: error=%.4f%%", ts.epoch, 100.0*ts.epochErr)
}

// Network holds all knowedlge of the network.
type Network struct {
	Layers         []*Layer
	EF             func(x, y float64) float64
	EFPrime        func(x, y float64) float64
	WeightInitFunc func() float64
	numout         int
	numlayers      int
	eta            float64
	momentum       float64
}

// NewNetwork construct the network based on the topology described in
// Add* and Set* options
func NewNetwork(options ...func(*Network)) *Network {
	n := &Network{
		eta: 1.0,
	}
	for _, option := range options {
		option(n)
	}
	// The last layer in the slice is the output layer
	n.numout = len(n.Layers[n.numlayers-1].Neurons)

	// Instantiate neurons and bias nodes
	var numcxns int
	for idx, layer := range n.Layers {
		if idx < n.numlayers-1 {
			numcxns = n.Layers[idx+1].numneurons
			for _, neuron := range layer.Neurons {
				neuron.Connections = make([]*Connection, numcxns)
				n.connectForwardLayer(neuron.Connections, idx+1)
			}
			if layer.Bias != nil {
				layer.Bias.Connections = make([]*Connection, numcxns)
				n.connectForwardLayer(layer.Bias.Connections, idx+1)
			}
		}
	}
	return n
}

// connectForwardLayer where cxns belong to a specific neuron
// and full connection to the foward layer is made
func (n *Network) connectForwardLayer(cxns []*Connection, layertoconnect int) {
	for idx := range cxns {
		cxns[idx] = &Connection{
			toNeuron: n.Layers[layertoconnect].Neurons[idx],
			Weight:   n.WeightInitFunc(),
		}
	}
}

func (n *Network) calcError(x, y float64) float64 {
	return n.EF(x, y)
}

// zeroNeuronValues to return the state of the network to accept new inputs.
// should be run after backpropagation when training or feed forward when predicting.
func (n *Network) zeroNeuronValues() {
	for _, layer := range n.Layers {
		for _, neuron := range layer.Neurons {
			neuron.Value = 0.0
			neuron.Delta = 0.0
		}
		if layer.Bias != nil {
			layer.Bias.Delta = 0.0
		}
	}
}

func (n *Network) String() string {
	s := "Network topology\n"
	for idx, layer := range n.Layers {
		s += fmt.Sprintf("Layer %v: %v with activation function %v\n",
			idx, layer.numneurons, layer.A.String())
	}
	return s
}

type Activator interface {
	F(x float64) float64
	FPrime(x float64) float64
	String() string
}

type Activate struct {
	forward  func(x float64) float64
	backward func(x float64) float64
	name     string
}

func (a Activate) F(x float64) float64 {
	return a.forward(x)
}

func (a Activate) FPrime(x float64) float64 {
	return a.backward(x)
}

func (a Activate) String() string {
	return a.name
}

var Binary Activator = Activate{
	forward: func(x float64) float64 {
		if x > 0.5 {
			return 1.0
		} else {
			return 0.0
		}
	},
	backward: func(x float64) float64 { return 1.0 },
	name:     "Binary",
}

var Linear Activator = Activate{
	forward:  func(x float64) float64 { return x },
	backward: func(x float64) float64 { return 1.0 },
	name:     "Linear",
}

var Sigmoid Activator = Activate{
	forward:  func(x float64) float64 { return 1 / (1 + math.Exp(-x)) },
	backward: func(x float64) float64 { return x * (1 - x) },
	name:     "Sigmoid",
}

var Tanh Activator = Activate{
	forward:  math.Tanh,
	backward: func(x float64) float64 { return 1 - math.Pow(x, 2) },
	name:     "Tanh",
}

type ErrFunc struct {
	forward  func(x, y float64) float64
	backward func(x, y float64) float64
}

func (e ErrFunc) F(x, y float64) float64 {
	return e.forward(x, y)
}

func (e ErrFunc) FPrime(x, y float64) float64 {
	return e.backward(x, y)
}

type ErrFuncer interface {
	F(x, y float64) float64
	FPrime(x, y float64) float64
}

var MSE ErrFuncer = ErrFunc{
	forward:  func(x, y float64) float64 { return 0.5 * math.Pow(x-y, 2) },
	backward: func(x, y float64) float64 { return x - y },
}
