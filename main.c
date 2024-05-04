#include <network.h>
#include <propagation.h>
#include <activation.h>
#include <training.h>
#include <mnist.h>

#include <stdio.h>

int main(int, char**) {
	struct training_set dataset;

	int ret = mnist_training_init(&dataset);
	if(ret == -1) return -1;

	struct layer *l1 = CREATE_HIDDEN_LAYER(dataset.hidden_nodes, 0, SIGMOID);
	struct layer *l2 = CREATE_HIDDEN_LAYER(dataset.hidden_nodes, 0, SIGMOID);

	printf("antwerp: establishing network connections\n");

	ret = initialise_layers(l1, l2, NULL);
	if(ret == -1) return -1;

	struct layer *input = create_input_layer(l1, dataset.input_nodes);
	if(input == NULL) return -1;

	struct layer *output = create_output_layer(l2, SIGMOID, dataset.output_nodes);
	if(input == NULL) return -1;
	
	struct network network = {
		.training_set = &dataset,
		.input = input,
		.hidden = l1,
		.output = output
	};

	ret = train(&network);
	if(ret == -1) return -1;

	return 0;
}
