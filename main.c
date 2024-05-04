#include <network.h>
#include <propagation.h>
#include <activation.h>
#include <mnist.h>

#include <stdio.h>

int main(int, char**) {
	struct mnist_training_data mnist_data;

	int ret = mnist_training_init(&mnist_data);
	if(ret == -1) return -1;

	struct layer *l1 = CREATE_HIDDEN_LAYER(3, 0, SIGMOID);
	struct layer *l2 = CREATE_HIDDEN_LAYER(3, 0, SIGMOID);
	struct layer *l3 = CREATE_HIDDEN_LAYER(3, 0, SIGMOID);
	struct layer *l4 = CREATE_HIDDEN_LAYER(3, 0, SIGMOID);
	struct layer *l5 = CREATE_HIDDEN_LAYER(3, 0, SIGMOID);

	printf("antwerp: establishing network connections\n");

	ret = initialise_layers(l1, l2, l3, l4, l5, NULL);
	if(ret == -1) return -1;

	struct layer *input = create_input_layer(l1, 6);
	if(input == NULL) return -1;

	struct layer *output = create_output_layer(l5, SIGMOID, 2);
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

	display_network(&network);

	return 0;
}
