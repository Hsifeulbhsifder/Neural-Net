#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <time.h>

#define sizeofArr(arr) ((sizeof(arr))/ sizeof(arr[0]))

//Learning rate, lower is slower, [0.0, 1.0]
#define ETA 1.0f

//Momentum, lower is less, [0.0, n]
#define ALPHA 0.15f

struct Connection{
	float weight;
	float dweight;
};

struct Neuron{
	Connection* connections;
	int numConnections;
	int index;
	float output;
	float gradient;
};

static float RandomWeight(){
	return (rand() / (float) RAND_MAX);
}

//Hyperbolic Tangent range [-1.0, 1.0]
static float TransferFunction(float x){
	return tanh(x);
}

//Hyperbolic Tangent derivative
static float TransferFunctionDerivative(float x){
	//tanh derivative
	return 1.0 - (tanh(x) * tanh(x));
}

void CreateNeuron(Neuron* neuron, int index, int numConnections){
	neuron->numConnections = numConnections;
	neuron->connections = (Connection*)malloc(sizeof(Connection) * numConnections);
	for(int i = 0; i < numConnections; i++){
		neuron->connections[i].weight = RandomWeight();
	}
	neuron->index = index;
}

float SumDOW(Neuron* neuron, Neuron* layer, int sizeOfLayer){
	float sum = 0;

	//Sum out contributions of the errors at the nodes we feed
	for(int i = 0; i < sizeOfLayer; i++){
		sum += neuron->connections[i].weight * layer[i].gradient;
	}

	return sum;
}

void NeuronFeedForward(Neuron* neuron, Neuron* previouslayer, int sizeOfPreviousLayer){
	float sum = 0.0f;

	//Sum previous layer's outputs
	for(int i = 0; i < sizeOfPreviousLayer; i++){
		sum += previouslayer[i].output * previouslayer[i].connections[neuron->index].weight;
	}

	neuron->output = TransferFunction(sum);
}

void CalcOutputGradients(Neuron* neuron, float target){
	float delta = target - neuron->output;
	neuron->gradient = delta * TransferFunctionDerivative(neuron->output);
}

void CalcHiddenGradients(Neuron* neuron, Neuron* nextLayer, int sizeOfNextLayer){
	float dow = SumDOW(neuron, nextLayer, sizeOfNextLayer);
	neuron->gradient = dow * TransferFunctionDerivative(neuron->output);
}

void UpdateInputWeights(Neuron* neuron, Neuron* previouslayer, int sizeOfPreviousLayer, float eta, float alpha){
	for(int i = 0; i < sizeOfPreviousLayer; i++){
		Neuron* prevNeuron = &previouslayer[i];
		float oldDweight = prevNeuron->connections[neuron->index].dweight;
		float newDweight = 
				//Individual input, magnified bythe gradient and train rate
				eta * prevNeuron->output * neuron->gradient
				//Also add momentum = a fraction of the previous delta weight
				+ alpha * oldDweight;

				prevNeuron->connections[neuron->index].dweight = newDweight;
				prevNeuron->connections[neuron->index].weight += newDweight;
	}
}

struct Net321{
	Neuron inputLayer[4];
	Neuron hiddenLayer0[3];
	Neuron outputLayer[2];
	float error;
	float recentAverageError;
	float recentAverageSmoothingError;
};

void CreateNet321(Net321* net){
	for(int i = 0; i < sizeofArr(net->inputLayer); i++){
		CreateNeuron(&(net->inputLayer[i]), i, sizeofArr(net->hiddenLayer0) - 1);
	}
	net->inputLayer[sizeofArr(net->inputLayer) - 1].output = 1.0f;
	
	for(int i = 0; i < sizeofArr(net->hiddenLayer0); i++){
		CreateNeuron(&(net->hiddenLayer0[i]), i, sizeofArr(net->outputLayer) - 1);
	}
	net->hiddenLayer0[sizeofArr(net->hiddenLayer0) - 1].output = 1.0f;

	for(int i = 0; i < sizeofArr(net->outputLayer); i++){
		CreateNeuron(&(net->outputLayer[i]), i,  0);
	}
	net->outputLayer[sizeofArr(net->outputLayer) - 1].output = 1.0f;
}

int GetNumWeights(Net321* net){
	int result = 0;
	result = (sizeofArr(net->inputLayer) - 1) * (sizeofArr(net->hiddenLayer0) - 1);
	result += (sizeofArr(net->hiddenLayer0) - 1) * (sizeofArr(net->outputLayer) - 1);
	return result;
}

void GetWeights(Net321* net, float* weights){
	int i = 0;
	for(i = 0; i < sizeofArr(net->inputLayer) - 1; i++){
		for(int j = 0; j < sizeofArr(net->hiddenLayer0) - 1; j++){
			weights[j + i * (sizeofArr(net->hiddenLayer0) - 1)] = net->inputLayer[i].connections[j].weight;
			printf("[0,%d,%d] = %f\n", i, j, weights[j + i * (sizeofArr(net->hiddenLayer0) - 1)]);
		}
	}

	for(int k = i + sizeofArr(net->hiddenLayer0), i = 0; i < sizeofArr(net->hiddenLayer0) - 1; i++){
		for(int j = 0; j < sizeofArr(net->outputLayer) - 1; j++){
			weights[j + k + i * (sizeofArr(net->outputLayer) - 1)] = net->hiddenLayer0[i].connections[j].weight;
			printf("[1,%d,%d] = %f\n", i, j, weights[j + k + i * (sizeofArr(net->outputLayer) - 1)]);
		}
	}
}

void SetWeights(Net321* net, float* weights){
	int i = 0;
	for(i = 0; i < sizeofArr(net->inputLayer) - 1; i++){
		for(int j = 0; j < sizeofArr(net->hiddenLayer0) - 1; j++){
			net->inputLayer[i].connections[j].weight = weights[j + i * (sizeofArr(net->hiddenLayer0) - 1)];
			printf("[0,%d,%d] = %f\n", i, j, net->inputLayer[i].connections[j].weight);
		}
	}

	for(int k = i + sizeofArr(net->hiddenLayer0), i = 0; i < sizeofArr(net->hiddenLayer0) - 1; i++){
		for(int j = 0; j < sizeofArr(net->outputLayer) - 1; j++){
			net->hiddenLayer0[i].connections[j].weight = weights[j + k + i * (sizeofArr(net->outputLayer) - 1)];
			printf("[1,%d,%d] = %f\n", i, j, net->inputLayer[i].connections[j].weight);
		}
	}
}

void FeedForward321(Net321* net, float* inputs, int sizeOfInputs){
	//Latch the inputs values into the neurons
	for(int i = 0; i < sizeOfInputs; i++){
		net->inputLayer[i].output = inputs[i];
	}

	//Forward propagate
	for(int i = 0; i < sizeofArr(net->hiddenLayer0) - 1; i++){
		NeuronFeedForward(&(net->hiddenLayer0[i]), net->inputLayer, sizeofArr(net->inputLayer));
	}

	for(int i = 0; i < sizeofArr(net->outputLayer) - 1; i++){
		NeuronFeedForward(&(net->outputLayer[i]), net->hiddenLayer0, sizeofArr(net->outputLayer));
	}
}

void BackPropagation321(Net321* net, float* targets, int sizeOfTargets){
	//Calculate overall net error (Root Mean Square Error of output neuron errors)
	Neuron* outputLayer = net->outputLayer;
	int outputLayerSize = sizeofArr(net->outputLayer) - 1;
	net->error = 0.0f;

	for(int i = 0; i < sizeOfTargets; i++){
		float delta = targets[i] - outputLayer[i].output;
		net->error += delta * delta;
	}
	net->error /= outputLayerSize; //Average
	net->error = sqrtf(net->error); //RMS

	//Implement a recent average measurement

	net->recentAverageError = (net->recentAverageError * net->recentAverageSmoothingError + net->error) 
						 / (net->recentAverageSmoothingError + 1.0f);

	//Calculate output layer gradients

	for(int i = 0; i < sizeOfTargets; i++){
		CalcOutputGradients(&outputLayer[i], targets[i]);
	}

	//Calculate gradients on hidden layers
	Neuron* hiddenLayer0 = net->hiddenLayer0;
	Neuron* nextLayer = net->outputLayer;
	for(int i = 0; i < sizeofArr(net->hiddenLayer0) - 1; i++){
		CalcHiddenGradients(&hiddenLayer0[i], nextLayer, sizeofArr(net->outputLayer) - 1);
	}

	//For all layers from outputs to first hidden layer, update connection weights
	{
		Neuron* layer = net->outputLayer;
		Neuron* prevLayer = net->hiddenLayer0;
		for(int i = 0; i < sizeofArr(net->outputLayer) - 1; i++){
			UpdateInputWeights(&layer[i], prevLayer, sizeofArr(net->hiddenLayer0), ETA, ALPHA);
		}
	}
	{
		Neuron* layer = net->hiddenLayer0;
		Neuron* prevLayer = net->inputLayer;
		for(int i = 0; i < sizeofArr(net->hiddenLayer0) - 1; i++){
			UpdateInputWeights(&layer[i], prevLayer, sizeofArr(net->inputLayer), ETA, ALPHA);
		}
	}
}

void GetResults321(Net321* net, float* results, int sizeOfResults){
	for(int i = 0; i < sizeOfResults; i++){
		results[i] = net->outputLayer[i].output;
	}
}

void Solve321(Net321* net, float* inputs, int sizeOfInputs, float* results, int sizeOfResults){
	FeedForward321(net, inputs, sizeOfInputs);
	GetResults321(net, results, sizeOfResults);
}

bool EpsilonEquals(float a, float b, float epsilon){
	return fabs(b - a) <= epsilon;
}
 
int main(int argc, char** argv){
	srand(time(NULL));


	Net321 net = {};
	CreateNet321(&net);

#if 0
	char* filename = "learningCase.lc";
	FILE* learningCase = fopen(filename, "r");

	if(learningCase){

		int numCases = 0;
		int numInputs = 0;
		int numOutputs = 0;
		fscanf(learningCase, "%d", &numCases);
		fscanf(learningCase, "%d", &numInputs);
		fscanf(learningCase, "%d", &numOutputs);

		float inputList[numInputs * numCases];
		float targetList[numOutputs * numCases];
		float results[numOutputs];

		for(int i = 0; i < numCases; i++){
			for(int j = 0; j < numInputs; j++){
				fscanf(learningCase, "%f", &inputList[j + i * numInputs]);
			}

			for(int j = 0; j < numOutputs; j++){
				fscanf(learningCase, "%f", &targetList[j + i * numOutputs]);
			}
		}

		fclose(learningCase);
		float avg = 0.0f;
		float lastValue = 0.0f;
		int convergeTime = 0;
		for(int i = 0; i < 20000000; i++){

			float sum = 0.0f;

			for(int j = 0; j < numCases; j++){
				float inputs[numInputs];
				float targets[numOutputs];

				for(int k = 0; k < numInputs; k++){
					inputs[k] = inputList[k + j * numCases];
					//printf("%f ", inputs[k]);
				}
				//printf("\n");

				for(int k = 0; k < numOutputs; k++){
					targets[k] = targetList[k + j * numCases];
					//printf("%f\n", targets[k]);
				}

				FeedForward321(&net, inputs, sizeofArr(inputs));
				BackPropagation321(&net, targets, sizeofArr(targets));
				GetResults321(&net, results, sizeofArr(results));

				lastValue = results[0];
				sum += net.recentAverageError;
				//printf("%f ||||| %f\n", net.recentAverageError, lastValue;
				
			}
			//printf("\n");

			avg = sum / numCases;
			convergeTime++;
			//printf("|||%d|||\n", convergeTime);

			if(avg <= 0.0001f){
				break;
			}
		}

		printf("Steps: %d Average: %f\n", convergeTime, avg);
		float weights[GetNumWeights(&net)];
		GetWeights(&net, weights);

		char* filename = "storedWeights.wt";
		FILE* storedWeights = fopen(filename, "w");
		for(int i = 0; i < sizeofArr(weights); i++){
			fprintf(storedWeights, "%f\n", weights[i]);
		}
		fclose(storedWeights);
	}
#else

		float weights[GetNumWeights(&net)];

		char* filename = "storedWeights.wt";
		FILE* storedWeights = fopen(filename, "r");
		for(int i = 0; i < sizeofArr(weights); i++){
			fscanf(storedWeights, "%f", &weights[i]);
		}

		SetWeights(&net, weights);

		fclose(storedWeights);

		float inputs[3];
		float results[1];

		for(int i = 0; i < sizeofArr(inputs); i++){
			scanf("%f", &inputs[i]);
		}

		Solve321(&net, inputs, sizeofArr(inputs), results, sizeofArr(results));
		printf("%f\n", results[0]);

#endif

}