package main

import (
	"fmt"
	"math"
	"time"
)

// Connection to forward layer
type Connection struct {
	Weight       float32
	weightChange float32
	toNeuron     *Neuron
	prevChange   float32
}

func (c *Connection) Copy() *Connection {
	return &Connection{
		Weight:     c.Weight,
		prevChange: c.prevChange,
	}
}

// Neuron in any layer
type Neuron struct {
	Value       float32
	Delta       float32
	Connections []*Connection
}

func (n *Neuron) Copy() *Neuron {
	connections := make([]*Connection, len(n.Connections))
	for idx, cxn := range n.Connections {
		connections[idx] = cxn.Copy()
	}
	return &Neuron{
		Value:       n.Value,
		Delta:       n.Delta,
		Connections: connections,
	}
}

// Bias is a neuron with a constant value of 1.0
// and a constant output of AF(1.0).
// This provides the constant term in the linear
// equation of layers where bias is included
type Bias struct {
	ONE         float32
	OUT         float32
	Delta       float32
	Connections []*Connection
}

func (b *Bias) Copy() *Bias {
	connections := make([]*Connection, len(b.Connections))
	for idx, cxn := range b.Connections {
		connections[idx] = cxn.Copy()
	}
	return &Bias{
		ONE:         b.ONE,
		OUT:         b.OUT,
		Delta:       b.Delta,
		Connections: connections,
	}
}

// Layer holds Neurons and Bias as well as functions
// for feedforward and back propigation step
type Layer struct {
	Neurons    []*Neuron
	Bias       *Bias
	A          Activator
	numneurons int
	shortcut   bool
}

func (l *Layer) Copy() *Layer {
	newNeurons := make([]*Neuron, len(l.Neurons))
	for idx, neuron := range l.Neurons {
		newNeurons[idx] = neuron.Copy()
	}
	var newBias *Bias
	if l.Bias != nil {
		newBias = l.Bias.Copy()
	}
	return &Layer{
		Neurons:    newNeurons,
		Bias:       newBias,
		A:          l.A,
		numneurons: l.numneurons,
		shortcut:   l.shortcut,
	}
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

// AddInputLayer to the network with n neurons
// Pass this function into NewNetwork to add a layer to the network.
func AddInputLayer(numneurons int, includeBias bool) func(*Network) {
	var bias *Bias
	if includeBias {
		bias = &Bias{ONE: 1.0,
			OUT: 1.0,
		}
	}
	neurons := make([]*Neuron, numneurons)
	for idx := range neurons {
		neurons[idx] = &Neuron{}
	}
	layer := &Layer{
		Neurons:    neurons,
		Bias:       bias,
		numneurons: numneurons,
	}
	return func(n *Network) {
		n.Layers = append(n.Layers, layer)
		n.numlayers++
	}
}

// AddInputLayer to the network with n neurons
// Pass this function into NewNetwork to add a layer to the network.
func AddOutputLayer(numneurons int, a Activator) func(*Network) {
	neurons := make([]*Neuron, numneurons)
	for idx := range neurons {
		neurons[idx] = &Neuron{}
	}
	return func(n *Network) {
		layer := &Layer{
			Neurons:    neurons,
			A:          a,
			numneurons: numneurons,
		}
		switch n.C.(type) {
		case CE:
			switch layer.A.(type) {
			case Sig:
				layer.shortcut = true
			}
		}
		n.Layers = append(n.Layers, layer)
		n.numlayers++
	}
}

// SettErrorFunction of the network.
// Pass this function into NewNetwork to set the error function of the network
func SetCostor(c Costor) func(*Network) {
	return func(n *Network) {
		n.C = c
		switch c.(type) {
		case CE:
			n.costDivisor = 1.0
		case MSE:
			n.costDivisor = float32(*outputs)
		}
	}
}

// SetLearningRate of the network. Range 0.0 to 1.0, default is 1.0
// Pass this function into NewNetwork to set the learning rate.
func SetLearningRate(eta float32) func(*Network) {
	return func(n *Network) {
		n.eta = eta
	}
}

// SetMomentum of the network. Range 0.0 to 1.0, default is 0.0
// Pass this function into NewNetwork to set the Momentum.
func SetMomentum(momentum float32) func(*Network) {
	return func(n *Network) {
		n.momentum = momentum
	}
}

// SetWeightInitFunc of the network.
// Pass this function into NewNetwork to set the weight initalization function.
func SetWeightInitFunc(weightPRNG WeightGenerator) func(*Network) {
	return func(n *Network) {
		n.WeightGenerator = weightPRNG
	}
}

// Network holds all knowedlge of the network.
type Network struct {
	Layers          []*Layer
	C               Costor
	WeightGenerator WeightGenerator
	costDivisor     float32
	numout          int
	numlayers       int
	eta             float32
	momentum        float32
}

func (n *Network) Copy() *Network {
	newLayers := make([]*Layer, n.numlayers)
	for idx, layer := range n.Layers {
		newLayers[idx] = layer.Copy()
	}
	newNet := &Network{
		Layers:          newLayers,
		C:               n.C,
		WeightGenerator: n.WeightGenerator,
		costDivisor:     n.costDivisor,
		numout:          n.numout,
		numlayers:       n.numlayers,
		eta:             n.eta,
		momentum:        n.momentum,
	}
	for layerIDX, layer := range newNet.Layers[0 : newNet.numlayers-1] {
		for _, neuron := range layer.Neurons {
			for cxnIDX, cxn := range neuron.Connections {
				cxn.toNeuron = newNet.Layers[layerIDX+1].Neurons[cxnIDX]
			}
		}
		if layer.Bias != nil {
			for cxnIDX, cxn := range layer.Bias.Connections {
				cxn.toNeuron = newNet.Layers[layerIDX+1].Neurons[cxnIDX]
			}
		}
	}
	return newNet
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
	n.numout = *outputs

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
			toNeuron:     n.Layers[layertoconnect].Neurons[idx],
			weightChange: 0.0,
			prevChange:   0.0,
			Weight:       n.WeightGenerator.Init(),
		}
	}
}

// zeroNeuronValues to return the state of the network to accept new inputs.
// should be run after backpropagation when training or feed forward when predicting.
func (n *Network) zeroValuesAndDeltas() {
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

func (n *Network) zeroValues() {
	for _, layer := range n.Layers {
		for _, neuron := range layer.Neurons {
			neuron.Value = 0.0
		}
	}
}

func (n *Network) String() string {
	s := "Network topology\n"
	deadweights := 0.0
	totalweights := 0.0
	var weights [][]float32
	for idx, layer := range n.Layers {
		weightLayer := make([]float32, 0)
		weights = append(weights, weightLayer)
		if layer.A != nil {
			s += fmt.Sprintf("Layer %v: %v neurons with activation function %v\n",
				idx, layer.numneurons, layer.A.String())
		} else {
			s += fmt.Sprintf("Input Layer %v: %v neurons\n",
				idx, layer.numneurons)
		}
		/*
			for _, neuron := range layer.Neurons {
				for _, cxn := range neuron.Connections {
					weights[idx] = append(weights[idx], cxn.Weight)
					if math.IsNaN(cxn.Weight) || math.IsInf(cxn.Weight, 0) {
						deadweights++
					}
					totalweights++
				}
			}
			if layer.Bias != nil {
				for _, cxn := range layer.Bias.Connections {
					weights[idx] = append(weights[idx], cxn.Weight)
					if math.IsNaN(cxn.Weight) || math.IsInf(cxn.Weight, 0) {
						deadweights++
					}
					totalweights++
				}
			}
		*/
	}
	wstats := weightStats(weights[:len(weights)-1])
	s += fmt.Sprintf("%v cost function\n", n.C)
	s += fmt.Sprintf("Percent dead weights: %.2f%%\n", 100.0*deadweights/totalweights)
	s += fmt.Sprintf("Weight mean by layer = %v\n", wstats["Mean"])
	s += fmt.Sprintf("Weight standard deviation = %v\n", wstats["StandardDeviation"])
	return s
}

func weightStats(weights [][]float32) map[string][]float32 {
	statMap := make(map[string][]float32)
	statMap["Mean"] = make([]float32, 0)
	statMap["StandardDeviation"] = make([]float32, 0)
	var weightCount float32
	for layerIDX, layer := range weights {
		statMap["Mean"] = append(statMap["Mean"], 0.0)
		for _, weight := range layer {
			statMap["Mean"][layerIDX] += weight
			weightCount++
		}
		statMap["Mean"][layerIDX] /= weightCount
	}
	for idx, layer := range weights {
		statMap["StandardDeviation"] = append(statMap["StandardDeviation"], 0.0)
		for _, weight := range layer {
			statMap["StandardDeviation"][idx] += float32(math.Pow(float64(weight-statMap["Mean"][idx]), 2))
		}
		statMap["StandardDeviation"][idx] /= weightCount
		statMap["StandardDeviation"][idx] = float32(math.Sqrt(float64(statMap["StandardDeviation"][idx])))
	}
	return statMap

}

type TrainStatus struct {
	sampCost  float32
	epochCost float32
	finalCost float32
	epoch     int
	epochTime time.Time
}

func (ts *TrainStatus) String() string {
	return fmt.Sprintf("Epoch %v: Cost=%e", ts.epoch, ts.epochCost)
}

func NewTrainStatus() *TrainStatus {
	return &TrainStatus{
		sampCost:  0.0,
		epochCost: 0.0,
		finalCost: 0.0,
		epoch:     1,
	}
}

type Activator interface {
	F(x float32) float32
	FPrime(x float32) float32
	String() string
}

type RL struct {
	forward  func(x float32) float32
	backward func(x float32) float32
	name     string
}

func (a RL) F(x float32) float32 {
	return a.forward(x)
}

func (a RL) FPrime(x float32) float32 {
	return a.backward(x)
}

func (a RL) String() string {
	return a.name
}

var Rectlin Activator = RL{
	forward: func(x float32) float32 {
		switch {
		case x >= 0.0:
			return x
		default:
			return 0.0
		}
	},
	backward: func(x float32) float32 {
		switch {
		case x > 0.0:
			return 1.0
		default:
			return 0.0
		}
	},
	name: "Rectified Linear",
}

type Lin struct {
	forward  func(x float32) float32
	backward func(x float32) float32
	name     string
}

func (a Lin) F(x float32) float32 {
	return a.forward(x)
}

func (a Lin) FPrime(x float32) float32 {
	return a.backward(x)
}

func (a Lin) String() string {
	return a.name
}

var Linear Activator = Lin{
	forward:  func(x float32) float32 { return x },
	backward: func(x float32) float32 { return 1.0 },
	name:     "Linear",
}

type Sig struct {
	forward  func(x float32) float32
	backward func(x float32) float32
	name     string
}

func (a Sig) F(x float32) float32 {
	return a.forward(x)
}

func (a Sig) FPrime(x float32) float32 {
	return a.backward(x)
}

func (a Sig) String() string {
	return a.name
}

var Sigmoid Activator = Sig{
	forward:  func(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) },
	backward: func(x float32) float32 { return x * (1 - x) },
	name:     "Sigmoid",
}

type TH struct {
	forward  func(x float32) float32
	backward func(x float32) float32
	name     string
}

func (a TH) F(x float32) float32 {
	return a.forward(x)
}

func (a TH) FPrime(x float32) float32 {
	return a.backward(x)
}

func (a TH) String() string {
	return a.name
}

var Tanh Activator = TH{
	forward:  func(x float32) float32 { return float32(math.Tanh(float64(x))) },
	backward: func(x float32) float32 { return 1 - float32(math.Pow(float64(x), 2)) },
	name:     "Tanh",
}

type Costor interface {
	F(o, t float32) float32
	FPrime(o, t float32) float32
	String() string
}

type MSE struct {
	forward  func(o, t float32) float32
	backward func(o, t float32) float32
	name     string
}

func (mse MSE) F(o, t float32) float32 {
	return mse.forward(o, t)
}

func (mse MSE) FPrime(o, t float32) float32 {
	return mse.backward(o, t)
}

func (mse MSE) String() string {
	return mse.name
}

var MeanSquared Costor = MSE{
	forward:  func(o, t float32) float32 { return 0.5 * float32(math.Pow(float64(o-t), 2.0)) },
	backward: func(o, t float32) float32 { return o - t },
	name:     "MSE",
}

type CE struct {
	forward  func(o, t float32) float32
	backward func(o, t float32) float32
	name     string
}

func (ce CE) F(o, t float32) float32 {
	return ce.forward(o, t)
}

func (ce CE) FPrime(o, t float32) float32 {
	return ce.backward(o, t)
}

func (ce CE) String() string {
	return ce.name
}

var CrossEntropy Costor = CE{
	forward:  func(o, t float32) float32 { return -1.0 * (safeLog(o)*t + safeLog(1.0-o)*(1.0-t)) },
	backward: func(o, t float32) float32 { return (o - t) },
	name:     "CrossEntropy",
}

func safeLog(x float32) float32 {
	return float32(math.Log(float64(x + epsillon)))
}

type WeightGen struct {
	init func() float32
	name string
}

func (w WeightGen) Init() float32 {
	return w.init()
}

func (w WeightGen) String() string {
	return w.name
}

type WeightGenerator interface {
	Init() float32
	String() string
}

func Normal(mean, std float32) WeightGenerator {
	return &WeightGen{
		init: func() float32 { return float32(pRNG.NormFloat64())*std + mean },
		name: fmt.Sprintf("Normal -- mean: %v and std: %v", mean, std),
	}
}
