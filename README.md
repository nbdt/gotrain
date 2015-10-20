Gotrain fits feedforward backpropigation neural networks to classification data.

## Usage

Create a toy dataset and train a multilayer perceptron network (MLP):

```
$ cat > ./xor.csv <<EOF
1,0,1
1,1,0
0,0,0
0,1,1
EOF
$ ./gotrain -datapath ./xor.csv MLP
```

For help on global flags and available artificial neuro networks (ANNs):

`$ ./gotrain -h`

For help on model specific flags:

`$ ./gotrain MLP -h`

## Future enhancements

* Support model persistence in [PMML](http://dmg.org/pmml/v4-1/GeneralStructure.html).
* Describe network topology outside of code.
* Mutate network topology during training to minimize neuron count.
* Optimize hyperparameters during training to converge faster.
* Implement more training algorithms.
