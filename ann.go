package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"
)

var modelANN = &Model{
	Train:       trainANNConcurent,
	Description: `ANN - train neural network with back propagation`,
}

var (
	momentum     float64
	learningRate float64
	errorThresh  float64
	maxEpochs    int
	numnets      int
)

func init() {
	modelANN.Flag.Float64Var(&momentum, "momentum", 0.0, "Momentum of network (default 0)")
	modelANN.Flag.Float64Var(&learningRate, "learningRate", 1.0, "Learning rate of the of network")
	modelANN.Flag.Float64Var(&errorThresh, "errorThresh", 0.001, "Minimum epoch error, end training")
	modelANN.Flag.IntVar(&maxEpochs, "maxEpochs", -1, "Maximum epoch training cycles")
	modelANN.Flag.IntVar(&numnets, "numnets", 1, "Number of networks to train concurrently")
}

type Result struct {
	epoch  int
	toterr float64
}

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
	AF         func(input float64) float64
	AFPrime    func(input float64) float64
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
		AF:         a.F,
		AFPrime:    a.FPrime,
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

// feedForward runs the feed forward pass based on inputs and produces outputs of the network
func (n *Network) feedForward(inputs []float64) (outputs []float64) {
	// Set value of input neurons
	for idx, inputneuron := range n.Layers[0].Neurons {
		inputneuron.Value = n.Layers[0].AF(inputs[idx])
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
		if idx < n.numlayers-1 {
			for _, neuron := range n.Layers[idx+1].Neurons {
				neuron.Value = n.Layers[idx+1].AF(neuron.Value)
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

// Execute prediction on input given a trained network
func (n *Network) Execute(input []float64) []float64 {
	output := n.feedForward(input)
	n.zeroNeuronValues()
	return output
}

// backPropagate adjusts the connection weights of the network based on the error, Delta,
// of each neuron computed during gradient decent
func (n *Network) backPropagate(targets []float64) {
	// Calculate delta of output layer
	outputlayer := n.Layers[n.numlayers-1]
	for idx, neuron := range outputlayer.Neurons {
		neuron.Delta = n.EFPrime(neuron.Value, targets[idx]) * outputlayer.AFPrime(neuron.Value)
	}

	for idx := n.numlayers - 2; idx >= 0; idx-- {
		layer := n.Layers[idx]
		for _, neuron := range layer.Neurons {
			for _, cxn := range neuron.Connections {
				neuron.Delta += cxn.Weight * cxn.toNeuron.Delta
			}
			neuron.Delta *= layer.AFPrime(neuron.Value)
		}
		if layer.Bias != nil {
			for _, cxn := range layer.Bias.Connections {
				layer.Bias.Delta += cxn.Weight * cxn.toNeuron.Delta
			}
			layer.Bias.Delta *= layer.AFPrime(layer.Bias.OUT)
		}
	}
	n.updateWeights()
	n.zeroNeuronValues()
}

// updateWeights based on precomputed deltas during back propagation
func (n *Network) updateWeights() {
	for _, layer := range n.Layers {
		for _, neuron := range layer.Neurons {
			for _, cxn := range neuron.Connections {
				momentumTerm := n.momentum * cxn.prevChange
				weightChange := n.eta*neuron.Value*cxn.toNeuron.Delta + momentumTerm
				cxn.prevChange = weightChange
				cxn.Weight -= weightChange
			}
		}
		if layer.Bias != nil {
			for _, cxn := range layer.Bias.Connections {
				momentumTerm := n.momentum * cxn.prevChange
				weightChange := n.eta*layer.Bias.ONE*cxn.toNeuron.Delta + momentumTerm
				cxn.prevChange = weightChange
				cxn.Weight -= weightChange
			}
		}
	}
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

func trainANN(data *TrainData, result chan<- *Network, done <-chan struct{}, wg *sync.WaitGroup) {
	defer wg.Done()
	if momentum < 0.0 || momentum > 1.0 {
		fmt.Fprintf(os.Stderr, "momentum [%v] is not between 0.0 and 1.0\n", momentum)
		os.Exit(2)
	}
	if learningRate < 0.0 || learningRate > 1.0 {
		fmt.Fprintf(os.Stderr, "learning rate [%v] is not between 0.0 and 1.0\n", learningRate)
		os.Exit(2)
	}
	// Define weight initialization function
	source := rand.NewSource(time.Now().UnixNano())
	pRNG := rand.New(source)
	WeightInitFunc := func() float64 {
		return 2 * (pRNG.Float64() - 0.5)
	}
	n := NewNetwork(SetLearningRate(learningRate),
		SetMomentum(momentum),
		SetErrorer(MSE),
		SetWeightInitFunc(WeightInitFunc),
		AddLayer(len(data.Input[0]), true, Sigmoid),
		AddLayer(50, true, Sigmoid),
		AddLayer(len(data.Output[0]), false, Sigmoid))

	// Train with stochastic gradient descent
	var (
		sgd        = NewSGD(len(data.Input), pRNG)
		choice     = 0
		sampledAll = false
		res        = &Result{epoch: 1}
		sampnum    = 0
		numsamp    = float64(len(data.Input))
		totalError = 0.0
	)
	for (res.toterr > errorThresh || !sampledAll) && (maxEpochs == -1 || res.epoch < maxEpochs) {
		select {
		case <-done:
			return
		default:
			if sampledAll {
				if *verbose {
					fmt.Fprintf(os.Stdout, "epoch %v error %E\n",
						res.epoch, res.toterr)
				}
				res.epoch++
				res.toterr = 0.0
			}
			choice, sampledAll = sgd.ChooseOne()
			h := n.feedForward(data.Input[choice])
			totalError = 0
			for idx := range h {
				// Sum average total error across all output neurons
				totalError += n.EF(h[idx], data.Output[choice][idx])
			}
			res.toterr = totalError / numsamp
			n.backPropagate(data.Output[choice])
			sampnum++
		}
	}
	fmt.Fprintln(os.Stdout, "Complete training of ANN")
	fmt.Fprintf(os.Stdout, "After final epoch %v error = %E\n", res.epoch, res.toterr)
	select {
	case result <- n:
	case <-done:
		return
	}
}

func trainANNConcurent(data *TrainData) (n Executor) {
	fmt.Fprintln(os.Stdout, "Beginning training of ANN")
	result := make(chan *Network)
	done := make(chan struct{})
	wg := sync.WaitGroup{}
	wg.Add(numnets)
	for n := 0; n < numnets; n++ {
		go trainANN(data, result, done, &wg)
	}
	network := <-result
	close(done)
	wg.Wait()
	close(result)
	return network
}

type Activator interface {
	F(x float64) float64
	FPrime(x float64) float64
}

type Activate struct {
	forward  func(x float64) float64
	backward func(x float64) float64
}

func (a Activate) F(x float64) float64 {
	return a.forward(x)
}

func (a Activate) FPrime(x float64) float64 {
	return a.backward(x)
}

var Linear Activator = Activate{
	forward:  func(x float64) float64 { return x },
	backward: func(x float64) float64 { return 1.0 },
}

var Sigmoid Activator = Activate{
	forward:  func(x float64) float64 { return 1 / (1 + math.Exp(-x)) },
	backward: func(x float64) float64 { return x * (1 - x) },
}

var Tanh Activator = Activate{
	forward:  math.Tanh,
	backward: func(x float64) float64 { return 1 - math.Pow(x, 2) },
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
