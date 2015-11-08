package main

/*
Implementation of stacked restricted boltzmann machines.

Constrastive Divergence (CD)

Recommendation to start with 1 round CD and increase as learning slows. EX:
CD1 ==> CD3 ==> CD5 ==> ... ==> learned weights

Modeling with real-valued data.
Visible units: linear with gaussian noise
Hidden units:  rectified linear units

Alternativly can use mean-feild approximation (0 - 255 pixel represents probabilty of pixel being on)

Fine tuning to improve generation:
1. Do a stochastic bottom-up pass
 -- adjust top-down weights of lower layers
2. Do a few iterations of sampling in the top level RBM
 -- do CD learning to update weights of top level RMB
3. Do a stochastic top-down pass
 -- adjust bottom-up weights
*/

var modelRBM = &Model{
	Train:       trainRBM,
	Description: `RBM -Stacked Restricted Boltzmann Machine with CD learning`,
}

type RBMExec struct {
	network *Network
}

func (rbm *RBMExec) Execute(input []float32) (output []float32) {
	output = input
	return output
}

func trainRBM(d *TrainData) (e Executor) {
	return &RBMExec{}
}
