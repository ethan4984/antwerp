#include <propagation.h>
#include <network.h>

#include <stdio.h>

static int forward_propagate_layer(struct layer*);

static int forward_propagate_layer(struct layer *layer) {
	if(layer == NULL) return -1;

	for(int i = 0; i < layer->n; i++) {
		struct perceptron *perceptron = &layer->perceptrons[i];

		perceptron->x = layer->bias;

		for(int j = 0; j < perceptron->n; j++) {
			struct signal *signal = &perceptron->signals[j];
	
			if(signal->emitter == NULL) continue;

			perceptron->x += signal->weight * signal->emitter->y;
		}

		if(layer->activation.function == NULL) continue;

		perceptron->y = layer->activation.function(perceptron->x);
	}

	return 0;
}

int forward_propagate(struct network *network) {
	if(network == NULL) return -1;

	struct layer *root = network->hidden;

	for(;;) {
		if(root == NULL) break;

		int ret = forward_propagate_layer(root);
		if(ret == -1) return -1;

		root = root->child;	
	}

	forward_propagate_layer(network->output);

	return 0;
}

int back_propagate(struct network *network, struct sample *sample) {
	if(network == NULL || sample == NULL) return -1;

	return 0;
}
