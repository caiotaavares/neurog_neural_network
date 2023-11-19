package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"
)

func main() {

	// Formaliza a matriz de treinamento
	// inputs, labels := makeInputsAndLabels("train.csv")
	inputs, labels := makeInputsAndLabels("treinamento.csv")

	// Define a arquitetura da nede neural
	config := neuralNetConfig{
		inputNeurons:  6, /*4*/
		outputNeurons: 5, /*3*/
		hiddenNeurons: 5, /*3*/
		numEpochs:     5000,
		learningRate:  0.3,
	}

	// Cria e  treina a rede neural
	network := createNeuralNetwork(config)
	if err := network.train(inputs, labels); err != nil {
		log.Fatal(err)
	}

}

// Define a arquitetura e os parâmetros de aprendizado da Rede Neural
type neuralNetConfig struct {
	inputNeurons  int     // Quantidade de neurônios de entrada
	outputNeurons int     // Quantidade de neurônios de saída
	hiddenNeurons int     // Quantidade de neurônios escondidos
	numEpochs     int     // Quantidade de "Epochs"
	learningRate  float64 // Taxa de aprendizado
}

// Contém as informações sobre o treinamento da rede neural
type neuralNet struct {
	config       neuralNetConfig // Configurações
	weightHidden *mat.Dense      // Matriz de pesos de entrada
	biasHidden   *mat.Dense      // Matriz de bias de entrada
	weightOut    *mat.Dense      // Matriz de pesos de saída
	biasOut      *mat.Dense      // Matriz de bias de saída
}

// Inicia uma nova rede neural
func createNeuralNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

// Implementação da sigmoid
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// implementação da sigmoid derivativa para o backpropagation
func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

// Treina a rede neural
func (nn *neuralNet) train(inputs, labels *mat.Dense) error {

	rand.Seed(time.Now().UnixNano())

	// Inicializa os pesos e bias
	weightInputHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
	biasInputHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
	weightHiddenOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
	biasHiddenOut := mat.NewDense(1, nn.config.outputNeurons, nil)

	weightInputHiddenRaw := weightInputHidden.RawMatrix().Data
	biasInputHiddenRaw := biasInputHidden.RawMatrix().Data
	weightHiddenOutRaw := weightHiddenOut.RawMatrix().Data
	biasHiddenOutRaw := biasHiddenOut.RawMatrix().Data

	// Atribui valores aleatórios aos pesos e bias
	for _, param := range [][]float64{
		weightInputHiddenRaw,
		biasInputHiddenRaw,
		weightHiddenOutRaw,
		biasHiddenOutRaw,
	} {
		for i := range param {
			param[i] = rand.Float64()*0.0002 - 0.0001
		}
	}

	fmt.Println("weightInputHidden\n", mat.Formatted(weightInputHidden, mat.Squeeze()))
	fmt.Println("biasInputHidden\n", mat.Formatted(biasInputHidden, mat.Squeeze()))
	fmt.Println("weightHiddenOut\n", mat.Formatted(weightHiddenOut, mat.Squeeze()))
	fmt.Println("biasHiddenOut\n", mat.Formatted(biasHiddenOut, mat.Squeeze()))

	// Saída da rede neural
	output := new(mat.Dense)

	// Usa o backpropagation para ajustar os pesos e bias
	if err := nn.backPropagate(
		inputs,
		labels,
		weightInputHidden,
		biasInputHidden,
		weightHiddenOut,
		biasHiddenOut,
		output); err != nil {
		return err
	}

	// // FIM DO TREINAMENTO: Implementa os elementos dentro da rede neural
	// nn.weightHidden = weightHidden
	// nn.biasHidden = biasHidden
	// nn.weightOut = weightOut
	// nn.biasOut = biasOut

	return nil
}

func (nn *neuralNet) backPropagate(inputs,
	labels,
	weightInputHidden,
	biasInputHidden,
	weightHiddenOut,
	biasHiddenOut,
	output *mat.Dense) error {

	// Loop através do número de epochs utilizando backpropagation
	for i := 0; i < nn.config.numEpochs; i++ {

		// FEEDFORWARD
		// Input -> hidden
		hiddenLayerInput := new(mat.Dense)
		fmt.Println("inputs\n", mat.Formatted(inputs, mat.Squeeze()))
		hiddenLayerInput.Mul(inputs, weightInputHidden)
		addBiasInputHidden := func(_, col int, v float64) float64 {
			return v + biasInputHidden.At(0, col)
		}
		hiddenLayerInput.Apply(addBiasInputHidden, hiddenLayerInput)
		// Aplicação da sigmoid
		InputHiddenLayerActivation := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 {
			return sigmoid(v)
		}
		InputHiddenLayerActivation.Apply(applySigmoid, hiddenLayerInput)

		// hidden -> output
		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(InputHiddenLayerActivation, weightHiddenOut)
		addBiasHiddenOut := func(_, col int, v float64) float64 {
			return v + biasHiddenOut.At(0, col)
		}
		outputLayerInput.Apply(addBiasHiddenOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// Backpropagation
		fmt.Println(labels.Dims())
		fmt.Println("labels:\n", mat.Formatted(labels, mat.Squeeze()))
		fmt.Println(output.Dims())
		fmt.Println("output:\n", mat.Formatted(output, mat.Squeeze()))
		networkError := new(mat.Dense)
		networkError.Sub(labels, output)
		// fmt.Println("Error:\n", mat.Formatted(networkError, mat.Squeeze()))
	}
	fmt.Println(output.Dims())
	fmt.Println("output\n", mat.Formatted(output, mat.Squeeze()))

	return nil
}

// Implementação do feed forward para previsão
// predict faz uma previsão com base em uma rede
// neural treinada.
func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {

	// Verifica se o valor neuralNet representa um modelo treinado.
	if nn.weightHidden == nil || nn.weightOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.biasHidden == nil || nn.biasOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	// Defina a saída da rede neural.
	output := new(mat.Dense)

	// Completa o processo de feed forward.
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.weightHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.biasHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.weightOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.biasOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

// Lê o CSV
func makeInputsAndLabels(fileName string) (*mat.Dense, *mat.Dense) {
	// Abra o arquivo do conjunto de dados.
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Crie um novo leitor de CSV lendo do arquivo aberto.
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	// Leia todos os registros CSV.
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// inputsData e labelsData irão conter todos os
	// valores de ponto flutuante que eventualmente serão
	// usados para formar matrizes.
	inputsData := make([]float64, 6*len(rawCSVData))
	labelsData := make([]float64, 1*len(rawCSVData))

	// Irá rastrear o índice atual dos valores da matriz.
	var inputsIndex int
	var labelsIndex int

	// Move sequentially through the rows to a slice of floating-point values.
	for idx, record := range rawCSVData {

		// Skip the header row.
		if idx == 0 {
			continue
		}

		// Iterate over the floating-point columns.
		for i, val := range record {

			// Convert the value to float.
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Convert the value to a floating-point.
			if i == 6 { // Assuming classe is the last column (index 6)
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Add to inputsData if relevant.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}
	inputs := mat.NewDense(len(rawCSVData), 6, inputsData)
	labels := mat.NewDense(len(rawCSVData), 1, labelsData)
	return inputs, labels
}

// Informações
// ShowModelInfo exibe informações sobre a rede neural após o treinamento.
func ShowModelInfo(network *neuralNet) {
	colorReset := "\033[0m"
	colorYellow := "\033[33m"

	fmt.Println(string(colorYellow))
	fmt.Println("Informações da Rede Neural APÓS O TREINAMENTO:")
	fmt.Printf("Arquitetura da Rede: %d -> %d -> %d\n", network.config.inputNeurons, network.config.hiddenNeurons, network.config.outputNeurons)
	fmt.Printf("Número de Épocas de Treinamento: %d\n", network.config.numEpochs)
	fmt.Printf("Taxa de Aprendizado: %f\n", network.config.learningRate)
	fmt.Println("Pesos e Vieses da Camada Oculta:")
	ShowWeightsAndBiases(network.weightHidden, network.biasHidden)
	fmt.Println("Pesos e Vieses da Camada de Saída:")
	ShowWeightsAndBiases(network.weightOut, network.biasOut)
	fmt.Println(string(colorReset))
}

// ShowPredictedInfo exibe informações sobre a rede neural após a função predict().
func ShowPredictedInfo(network *neuralNet, predictions *mat.Dense) {
	colorReset := "\033[0m"
	colorGreen := "\033[32m"

	fmt.Println(string(colorGreen))
	fmt.Println("Informações da Rede Neural APÓS O Predict():")
	fmt.Printf("Arquitetura da Rede: %d -> %d -> %d\n", network.config.inputNeurons, network.config.hiddenNeurons, network.config.outputNeurons)
	fmt.Printf("Número de Épocas de Treinamento: %d\n", network.config.numEpochs)
	fmt.Printf("Taxa de Aprendizado: %f\n", network.config.learningRate)
	fmt.Println("Pesos e Vieses da Camada Oculta:")
	ShowWeightsAndBiases(network.weightHidden, network.biasHidden)
	fmt.Println("Pesos e Vieses da Camada de Saída:")
	ShowWeightsAndBiases(network.weightOut, network.biasOut)
	fmt.Println(string(colorReset))
}

// ShowWeightsAndBiases exibe os pesos e vieses de uma camada da rede neural.
func ShowWeightsAndBiases(weights, biases *mat.Dense) {
	fmt.Println("Pesos:")
	fmt.Println(mat.Formatted(weights, mat.Squeeze()))
	fmt.Println("Vieses:")
	fmt.Println(mat.Formatted(biases, mat.Squeeze()))
}
