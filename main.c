#include <network.h>
#include <propagation.h>
#include <activation.h>
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
		.input = input,
		.hidden = l1,
		.output = output
	};

	for(int i = 0; i < input->n; i++) input->perceptrons[i].y = i;

	printf("antwerp: forward propagating\n");

	ret = forward_propagate(&network);
	if(ret == -1) {
		printf("antwerp: failure during forward propagation\n");
		return -1;
	}

	return 0;
}
