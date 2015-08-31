package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
)

var (
	datapath   = flag.String("datapath", "", "path to train data csv")
	exportpath = flag.String("exportpath", "", "path to export model after training")
	cvsettings = flag.String("cvsettings", "", "path to cross validation settings file")
	classes    = flag.Int("classes", 1, "number of training classes")
	verbose    = flag.Bool("verbose", false, "verbose output")
)

type Executor interface {
	Execute(data []float64) []float64
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

func (c *Model) Usage() {
	fmt.Fprintf(os.Stderr, "usage: %s [MODEL OPTION]...\n\n", c.Name())
	c.Flag.PrintDefaults()
	os.Exit(2)
}

func (m *Model) SetExecutor(e Executor) {
	m.Predict = e
}

// models list the available models and usage.
// The order here is the order in which they are printed by 'gotrain help'.
var models = []*Model{
	modelANN,
}

func main() {
	flag.Usage = usage
	flag.Parse()
	args := flag.Args()

	data, err := ReadCSV(*datapath, *classes)
	if err != nil {
		fmt.Printf("%s\n", err)
		os.Exit(1)
	}
	for _, model := range models {
		if model.Name() == args[0] {
			model.Flag.Usage = func() { model.Usage() }
			model.Flag.Parse(args[1:])
			trained := model.Train(data)
			model.SetExecutor(trained)
			if *exportpath != "" {
				fmt.Fprintln(os.Stderr, "Model export not yet implemented")
			}
		}
	}
	os.Exit(0)
}

func usage() {
	fmt.Fprintf(os.Stderr, "%s\n\n", "usage: godo [OPTION]... model [MODEL OPTION]...\n")
	fmt.Fprintln(os.Stderr, "OPTION flags include:")
	flag.PrintDefaults()
	fmt.Fprintln(os.Stderr)
	for _, model := range models {
		fmt.Fprintf(os.Stderr, "%s\n", model.Description)
	}
	fmt.Fprintln(os.Stderr, "\nFor help on model options use godo MODEL -help")
}
