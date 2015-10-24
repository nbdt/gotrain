package main

import (
	"fmt"
	"os"
	"sync"
)

var modelMLP = &Model{
	Train:       trainMLPConcurent,
	Description: `MLP - train feed-forward neural network with back propagation`,
}

var (
	momentum     float64
	learningRate float64
	errorThresh  float64
	maxEpochs    int
	numnets      int
	batchsize    int
)

func init() {
	modelMLP.Flag.Float64Var(&momentum, "momentum", 0.0, "Momentum of network (default 0)")
	modelMLP.Flag.Float64Var(&learningRate, "learningRate", 1.0, "Learning rate of the of network")
	modelMLP.Flag.Float64Var(&errorThresh, "errorThresh", 0.001, "Minimum epoch error, end training")
	modelMLP.Flag.IntVar(&maxEpochs, "maxEpochs", -1, "Maximum epoch training cycles")
	//for model parallelism
	modelMLP.Flag.IntVar(&numnets, "numnets", 1, "Number of networks to train concurrently")
	//for data parallelism
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

type MLP interface {
	feedForward(inputs []float64) (outputs []float64)
	backPropagate(targets []float64)
	zeroNetwork()
	calcError(x, y float64) float64
	Execute(inputs []float64) (outputs []float64)
	String() string
}

// NewMLP constructs the convolutional neuro network based on the topology described in
// Add* and Set* options
func NewMLP(insize, outsize int) MLP {
	// Define weight initialization function
	n := NewNetwork(SetLearningRate(learningRate),
		SetMomentum(momentum),
		SetErrorer(MSE),
		SetWeightInitFunc(Normal(0.0, 0.1)),
		AddInputLayer(insize),
		AddLayer(100, false, Tanh),
		AddLayer(50, false, Tanh),
		AddLayer(outsize, false, Sigmoid))
	return n
}

// feedForward runs the feed forward pass based on inputs and produces outputs of the network
func (n *Network) feedForward(inputs []float64) (outputs []float64) {
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
		//Dont apply the activation function to the input layer
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
func (n *Network) backPropagate(targets []float64) {
	// Calculate delta of output layer
	outputlayer := n.Layers[n.numlayers-1]
	for idx, neuron := range outputlayer.Neurons {
		neuron.Delta = n.EFPrime(neuron.Value, targets[idx]) * outputlayer.A.FPrime(neuron.Value)
	}

	for idx := n.numlayers - 2; idx >= 1; idx-- {
		layer := n.Layers[idx]
		for _, neuron := range layer.Neurons {
			for _, cxn := range neuron.Connections {
				neuron.Delta += cxn.Weight * cxn.toNeuron.Delta
			}
			neuron.Delta *= layer.A.FPrime(neuron.Value)
		}
		if layer.Bias != nil {
			for _, cxn := range layer.Bias.Connections {
				layer.Bias.Delta += cxn.Weight * cxn.toNeuron.Delta
			}
			layer.Bias.Delta *= layer.A.FPrime(layer.Bias.OUT)
		}
	}
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
	n.zeroNetwork()
}

// Execute prediction on input given a trained network
func (n *Network) Execute(input []float64) []float64 {
	output := n.feedForward(input)
	n.zeroNetwork()
	return output
}

type MLPResult struct {
	net         MLP
	trainStatus *TrainStatus
	resp        chan bool
}

func trainMLP(data *TrainData, result chan<- MLPResult, quit chan struct{}, wg *sync.WaitGroup) {
	defer wg.Done()
	n := NewMLP(len(data.Input[0]), len(data.Output[0]))
	fmt.Fprintf(os.Stdout, "%v\n", n)

	ts := &TrainStatus{
		epoch:    1.0,
		epochErr: 0.0,
	}
	// Train with stochastic gradient descent
	var (
		sgd        = NewSGD(len(data.Input), pRNG)
		choice     = 0
		sampledAll = false
		numsamp    = float64(len(data.Input))
		sampError  = 0.0
	)
	for {
		select {
		case <-quit:
			return
		default:
			if sampledAll {
				if *verbose {
					fmt.Fprintf(os.Stdout, "%v\n", ts)
				}
				res := &MLPResult{
					net:         n,
					trainStatus: ts,
					resp:        make(chan bool),
				}
				select {
				case <-quit:
					return
				case result <- *res:
					done := <-res.resp
					close(res.resp)
					if done {
						return
					}
					ts.epoch++
					ts.epochErr = 0.0
				}
			}
			choice, sampledAll = sgd.ChooseOne()
			h := n.feedForward(data.Input[choice])
			sampError = 0.0
			for idx := range h {
				// Sum average total error across all output neurons
				sampError += n.calcError(h[idx], data.Output[choice][idx])
			}
			sampError /= float64(len(h))
			ts.epochErr += sampError / numsamp
			n.backPropagate(data.Output[choice])
		}
	}
}

func trainMLPConcurent(data *TrainData) (n Executor) {
	// define a keeper incase training never drops below the defined
	// error threshold. if we break on max epochs, we keep the network
	// with the lowest epoch error.
	keeper := MLPResult{
		trainStatus: &TrainStatus{
			epoch:    0,
			epochErr: 1.0,
		}}

	fmt.Fprintf(os.Stdout, "Beginning training of ANN until max epoch: %v or error threshold: %v\n",
		maxEpochs, errorThresh)
	result := make(chan MLPResult)
	quit := make(chan struct{})
	wg := &sync.WaitGroup{}
	wg.Add(numnets)
	for n := 0; n < numnets; n++ {
		go trainMLP(data, result, quit, wg)
	}
	// Either one network trains below the error threshold
	// or all networks reach the epoch limit.
	// If the epoch limit is reached by all, return the keeper,
	// that is the network with the lowest insample error.
	var res MLPResult
	var trialsFinished int
TrainLoop:
	for {
		select {
		case res = <-result:
			if res.trainStatus.epochErr < keeper.trainStatus.epochErr {
				keeper = res
			}
			if res.trainStatus.epochErr < errorThresh || isStopEarly() {
				close(quit)
				res.resp <- true
				break TrainLoop
			} else if res.trainStatus.epoch == maxEpochs {
				res.resp <- true
				trialsFinished++
				if trialsFinished == numnets {
					close(quit)
					break TrainLoop
				}
			} else {
				res.resp <- false
			}
		}
	}
	wg.Wait()
	close(result)
	if keeper.trainStatus.epochErr < res.trainStatus.epochErr {
		res = keeper
	}
	fmt.Fprintln(os.Stdout, "Complete training of ANN")
	fmt.Fprintf(os.Stdout, "Keeping: %v\n", res.trainStatus)
	return res.net
}
