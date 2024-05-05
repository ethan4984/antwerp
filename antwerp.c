#include <antwerp.h>

#include <network.h>
#include <propagation.h>
#include <training.h>
#include <mnist.h>

#include <stdio.h>

int main(int, char**) {
	struct training_set dataset;

	int ret = mnist_training_init(&dataset);
	if(ret == -1) return -1;

	struct network network;

	struct layer *l1 = CREATE_HIDDEN_LAYER(&network, dataset.hidden_nodes, 0, SIGMOID);

	printf("antwerp: establishing network connections\n");

	ret = initialise_layers(&network, l1, NULL);
	if(ret == -1) return -1;

	struct layer *input = create_input_layer(&network, l1, dataset.input_nodes);
	if(input == NULL) return -1;

	struct layer *output = create_output_layer(&network, l1, SIGMOID, dataset.output_nodes);
	if(output == NULL) return -1;

	network.training_set = &dataset;
	network.learning_rate = 0.005;

	ret = train(&network);
	if(ret == -1) return -1;

	return 0;
}
