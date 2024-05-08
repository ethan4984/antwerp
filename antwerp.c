#include <antwerp.h>
#include <network.h>
#include <propagation.h>
#include <mnist.h>

#include <stdio.h>

static int network_mnist(void);
static int network_test(void);

static int network_mnist(void) {
	struct training_set dataset;

	int ret = mnist_training_init(&dataset);
	if(ret == -1) return -1;

	struct network network = {
		.training_set = &dataset,
		.learning_rate = 0.05
	};

	ret = network_init(&network, 3, SIGMOID, dataset.input_nodes, 0,
			dataset.hidden_nodes, 0,
			dataset.output_nodes, 0);
	if(ret == -1) return -1;

	ret = train(&network);
	if(ret == -1) return -1;

	return 0;
}

static int network_test(void) { 
	struct training_set dataset;

	dataset.input_nodes = 2;
	dataset.hidden_nodes = 3;
	dataset.output_nodes = 2;

	struct network network = {
		.training_set = &dataset,
		.learning_rate = 0.5
	};

	int ret = network_init(&network, 4, SIGMOID, dataset.input_nodes, 0,
			dataset.hidden_nodes, 0,
			dataset.hidden_nodes, 0,
			dataset.output_nodes, 0);
	if(ret == -1) return -1;

	network.a[0][0] = 0.10;
	network.a[0][1] = -0.20;

	network.weights[1][0][0] = 0.40;
	network.weights[1][0][1] = -0.20;
	network.weights[1][1][0] = -0.10;
	network.weights[1][1][1] = 0.90;
	network.weights[1][2][0] = 0.80;
	network.weights[1][2][1] = -0.10;

	network.weights[2][0][0] = 0.20;
	network.weights[2][0][1] = -0.10;
	network.weights[2][0][2] = 0.50;
	network.weights[2][1][0] = 0.8;
	network.weights[2][1][1] = 0.9;
	network.weights[2][1][3] = 0.1;
	network.weights[2][2][0] = -0.6;
	network.weights[2][2][1] = 0.45;
	network.weights[2][2][2] = 0.56;

	network.weights[3][0][0] = 0.8;
	network.weights[3][0][1] = -0.5;
	network.weights[3][0][2] = 0.1;
	network.weights[3][1][0] = 0.9;
	network.weights[3][1][1] = 0.19;
	network.weights[3][1][2] = 0.30;

	network.expected[0] = 0.0;
	network.expected[1] = 1.0;

	ret = forward_propagate(&network);
	if(ret == -1) return -1;

	network_display(&network);

	for(int i = 0; i < 100; i++) backward_propagate(&network);

	ret = forward_propagate(&network);
	if(ret == -1) return -1;

	network_display(&network);

	return 0;
}

int main(int, char**) {
	return network_mnist();
	return network_test();
}
