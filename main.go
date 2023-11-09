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

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func main() {

	// Formaliza a matriz de treinamento
	inputs, labels := makeInputsAndLabels("train.csv")

	// Define a arquitetura da nede neural
	config := neuralNetConfig{
		inputNeurons:  4,
		outputNeurons: 3,
		hiddenNeurons: 3,
		numEpochs:     5000,
		learningRate:  0.3,
	}

	// Treina a rede neural
	network := createNeuralNetwork(config)
	if err := network.train(inputs, labels); err != nil {
		log.Fatal(err)
	}

	// Mostra as informações sobre a network após o treinamento
	ShowModelInfo(network)

	// Formaliza a matriz de teste
	testInputs, testLabels := makeInputsAndLabels("test.csv")

	// Realiza as predições usando o modelo treinado (network)
	predictions, err := network.predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// Mostra as informações sobre a network após a predição
	ShowPredictedInfo(network, predictions)

	// Printa a matriz de confusão
	PrintConfusionMatrix(predictions, testLabels)

	// Cálculo da acurácia do modelo
	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ {

		// Camada
		labelRow := mat.Row(nil, i, testLabels)
		var prediction int
		for idx, label := range labelRow {
			if label == 1.0 {
				prediction = idx
				break
			}
		}

		// Acumula os verdadeiro positivo/negativo
		if predictions.At(i, prediction) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	// Cáluclo da acurácia
	accuracy := float64(truePosNeg) / float64(numPreds)

	// Mostra a acurácia
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)
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

// Treina a rede neural usando backpropagation
func (nn *neuralNet) train(inputs, labels *mat.Dense) error {

	// Inicializa os pesos e bias
	randSource := rand.NewSource(time.Now().UnixNano())
	randGem := rand.New(randSource)

	weightHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
	biasHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
	weightOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
	biasOut := mat.NewDense(1, nn.config.outputNeurons, nil)

	weightHiddenRaw := weightHidden.RawMatrix().Data
	biasHiddenRaw := biasHidden.RawMatrix().Data
	weightOutRaw := weightOut.RawMatrix().Data
	biasOutRaw := biasOut.RawMatrix().Data

	// Atribui valores aleatórios aos pesos e bias
	for _, param := range [][]float64{
		weightHiddenRaw,
		biasHiddenRaw,
		weightOutRaw,
		biasOutRaw,
	} {
		for i := range param {
			param[i] = randGem.Float64()
		}
	}

	// Saída da rede neural
	output := new(mat.Dense)

	// Usa o backpropagation para ajustar os pesos e bias
	if err := nn.backPropagate(inputs, labels, weightHidden, biasHidden, weightOut, biasOut, output); err != nil {
		return err
	}

	// Implementa os elementos dentro da rede neural
	nn.weightHidden = weightHidden
	nn.biasHidden = biasHidden
	nn.weightOut = weightOut
	nn.biasOut = biasOut

	return nil
}

func (nn *neuralNet) backPropagate(inputs, labels, weightHidden, biasHidden, weightOut, biasOut, output *mat.Dense) error {

	// Loop através do número de epochs utilizando backpropagation
	for i := 0; i < nn.config.numEpochs; i++ {

		// Completa o processo de feedforward
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(inputs, weightHidden)
		addBiasHidden := func(_, col int, v float64) float64 {
			return v + biasHidden.At(0, col)
		}
		// Aplica a função de "addBiasHidden" em cada elemento de "hiddenLayerInput"
		// e aplicando o resultado no recebedor
		hiddenLayerInput.Apply(addBiasHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 {
			return sigmoid(v)
		}
		// Aplica a função de "applySigmoid" em cada elemento de "hiddenLayerActivations"
		// e aplicando o resultado no recebedor
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, weightOut)
		addBiasOut := func(_, col int, v float64) float64 {
			return v + biasOut.At(0, col)
		}
		outputLayerInput.Apply(addBiasOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// Completa a backpropagation.
		networkError := new(mat.Dense)
		networkError.Sub(labels, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 {
			return sigmoidPrime(v)
		}
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)
		//
		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, weightOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// Ajusta os parâmetros
		weightOutAdj := new(mat.Dense)
		weightOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		weightOutAdj.Scale(nn.config.learningRate, weightOutAdj)
		weightOut.Add(weightOut, weightOutAdj)

		biasOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		biasOutAdj.Scale(nn.config.learningRate, biasOutAdj)
		biasOut.Add(biasOut, biasOutAdj)

		weightHiddenAdj := new(mat.Dense)
		weightHiddenAdj.Mul(inputs.T(), dHiddenLayer)
		weightHiddenAdj.Scale(nn.config.learningRate, weightHiddenAdj)
		weightHidden.Add(weightHidden, weightHiddenAdj)

		biasHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		biasHiddenAdj.Scale(nn.config.learningRate, biasHiddenAdj)
		biasHidden.Add(biasHidden, biasHiddenAdj)
	}

	return nil
}

// sumAlongAxis soma uma matriz ao longo de uma dimensão específica,
// preservando a outra dimensão.
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
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

	// Complete o processo de feed forward.
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
	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	// Irá rastrear o índice atual dos valores da matriz.
	var inputsIndex int
	var labelsIndex int

	// Movimenta sequencialmente as linhas para uma fatia de valores de ponto flutuante.
	for idx, record := range rawCSVData {

		// Pula a linha de cabeçalho.
		if idx == 0 {
			continue
		}

		// Percorre as colunas de ponto flutuante.
		for i, val := range record {

			// Convrte o valor para float
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Converta o valor para um ponto flutuante.
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Adicione ao dadosLabels se for relevante.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}
	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)
	return inputs, labels
}

// ConfusionMatrix calcula a matriz de confusão com base nas previsões do modelo e nos rótulos reais.
func ConfusionMatrix(predictions, actualLabels *mat.Dense) (int, int, int, int) {
	numSamples, _ := predictions.Dims()
	VP, VN, FP, FN := 0, 0, 0, 0

	for i := 0; i < numSamples; i++ {
		predictedRow := mat.Row(nil, i, predictions)
		actualRow := mat.Row(nil, i, actualLabels)

		maxIndexPredicted := floats.MaxIdx(predictedRow)
		maxIndexActual := floats.MaxIdx(actualRow)

		if maxIndexPredicted == maxIndexActual {
			if maxIndexActual == 1 { // Verdadeiro Positivo
				VP++
			} else { // Verdadeiro Negativo
				VN++
			}
		} else {
			if maxIndexPredicted == 1 { // Falso Positivo
				FP++
			} else { // Falso Negativo
				FN++
			}
		}
	}

	return VP, VN, FP, FN
}

// PrintConfusionMatrix calcula e imprime a matriz de confusão com base nas previsões do modelo e nos rótulos reais.
func PrintConfusionMatrix(predictions, actualLabels *mat.Dense) {
	VP, VN, FP, FN := ConfusionMatrix(predictions, actualLabels)

	fmt.Println("Matriz de Confusão:")
	fmt.Printf("Verdadeiro Positivo (VP): %d\n", VP)
	fmt.Printf("Verdadeiro Negativo (VN): %d\n", VN)
	fmt.Printf("Falso Positivo (FP): %d\n", FP)
	fmt.Printf("Falso Negativo (FN): %d\n", FN)
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
	colorBlue := "\033[34m"

	fmt.Println(string(colorGreen))
	fmt.Println("Informações da Rede Neural APÓS O Predict():")
	fmt.Printf("Arquitetura da Rede: %d -> %d -> %d\n", network.config.inputNeurons, network.config.hiddenNeurons, network.config.outputNeurons)
	fmt.Printf("Número de Épocas de Treinamento: %d\n", network.config.numEpochs)
	fmt.Printf("Taxa de Aprendizado: %f\n", network.config.learningRate)
	fmt.Println("Pesos e Vieses da Camada Oculta:")
	ShowWeightsAndBiases(network.weightHidden, network.biasHidden)
	fmt.Println("Pesos e Vieses da Camada de Saída:")
	ShowWeightsAndBiases(network.weightOut, network.biasOut)

	fmt.Println(string(colorBlue))
	fmt.Println("Previsões do Modelo:")
	ShowPredictions(predictions)
	fmt.Println(string(colorReset))
}

// ShowWeightsAndBiases exibe os pesos e vieses de uma camada da rede neural.
func ShowWeightsAndBiases(weights, biases *mat.Dense) {
	fmt.Println("Pesos:")
	fmt.Println(mat.Formatted(weights, mat.Squeeze()))
	fmt.Println("Vieses:")
	fmt.Println(mat.Formatted(biases, mat.Squeeze()))
}

// ShowPredictions exibe as previsões do modelo.
func ShowPredictions(predictions *mat.Dense) {
	fmt.Println("Previsões:")
	fmt.Println(mat.Formatted(predictions, mat.Squeeze()))
}
