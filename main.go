package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"runtime/pprof"
	"strings"
	"time"
)

var (
	cpuprofile       = flag.String("cpuprofile", "", "write cpu profile to file")
	traindatapath    = flag.String("traindatapath", "", "path to train data csv")
	validatedatapath = flag.String("validatedatapath", "", "path to train data csv")
	testdatapath     = flag.String("testdatapath", "", "path to train data csv")
	exportpath       = flag.String("exportpath", "", "path to export model after training")
	outputs          = flag.Int("outputs", 0, "number of training outputs")
	verbose          = flag.Bool("verbose", false, "verbose output")
)

type Executor interface {
	Execute(data []float32) []float32
}

type Model struct {
	// Short is the short description shown in the 'go help' output.
	Description string

	// Flag is a set of flags specific to this command.
	Flag flag.FlagSet

	// Predict with trained model give a set of inputs produce outputs
	Predict Executor

	// Train on train data
	Train func(d *TrainData) Executor
}

// Name returns the command's name: the first word in the usage line.
func (m *Model) Name() string {
	name := m.Description
	i := strings.Index(name, " ")
	if i >= 0 {
		name = name[:i]
	}
	return name
}

func (m *Model) String() string {
	return m.Name()
}

func (m *Model) Usage() {
	fmt.Fprintf(os.Stderr, "usage: %s [MODEL OPTION]...\n\n", m.Name())
	m.Flag.PrintDefaults()
	fmt.Fprintln(os.Stderr)
	os.Exit(2)
}

func (m *Model) SetExecutor(e Executor) {
	m.Predict = e
}

// models list the available models and usage.
// The order here is the order in which they are printed by 'gotrain help'.
var models = []*Model{
	modelMLP,
	modelRBM,
}

var pRNG *rand.Rand

func init() {
	source := rand.NewSource(time.Now().UnixNano())
	pRNG = rand.New(source)
}

func main() {
	flag.Usage = usage
	flag.Parse()
	args := flag.Args()
	didSomethingHappen := false
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	if *outputs == 0 {
		fmt.Fprintln(os.Stderr, "Must specify number of outputs")
		os.Exit(2)
	}

	for _, model := range models {
		if model.Name() == args[0] {
			didSomethingHappen = true
			model.Flag.Usage = func() { model.Usage() }
			model.Flag.Parse(args[1:])
			data, err := ReadCSV(*traindatapath, *outputs)
			if err != nil {
				fmt.Printf("%s\n", err)
				os.Exit(1)
			}
			trained := model.Train(data)
			model.SetExecutor(trained)
			if *exportpath != "" {
				fmt.Fprintln(os.Stderr, "Model export not yet implemented")
			}
			if *validatedatapath != "" {
				doValidation(model.Predict)
			}
			if *testdatapath != "" {
				doTest(model.Predict)
			}
		}
	}
	if !didSomethingHappen {
		fmt.Fprintf(os.Stderr, "Model %v not recognized, models: %v are supported\n",
			args[0], models)
	}
	fmt.Fprintln(os.Stderr, "Exiting cleanly")
}

func usage() {
	fmt.Fprintf(os.Stderr, "%s\n\n", "usage: gotrain [OPTION]... model [MODEL OPTION]...")
	fmt.Fprintln(os.Stderr, "OPTION flags include:")
	flag.PrintDefaults()
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr, "Available models include:")
	for _, model := range models {
		fmt.Fprintf(os.Stderr, "%s\n", model.Description)
	}
	fmt.Fprintln(os.Stderr, "\nFor help on model options use train MODEL -h\n")
}
