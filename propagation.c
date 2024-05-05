#include <propagation.h>
#include <network.h>

#include <stdio.h>

static int forward_propagate_layer(struct layer*);

static int backward_propagate_layer(struct layer*, struct sample*);
static double cost_gradient(struct perceptron*, struct signal*, struct sample*);

static int forward_propagate_layer(struct layer *layer) {
	if(layer == NULL) return -1;

	for(int i = 0; i < layer->n; i++) {
		struct perceptron *perceptron = &layer->perceptrons[i];

		perceptron->z = layer->bias;

		for(int j = 0; j < perceptron->n; j++) {
			struct signal *signal = &perceptron->signals[j];
	
			if(signal->emitter == NULL) continue;

			perceptron->z += signal->weight * signal->emitter->a;
		}

		if(layer->activation.function == NULL) continue;

		perceptron->a = layer->activation.function(perceptron->z);
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

static double cost_gradient(struct perceptron *perceptron,
		struct signal *signal, struct sample *sample) {
	if(perceptron == NULL || signal == NULL ||
			sample == NULL || perceptron->layer == NULL) return NAN;
	
	struct layer *layer = perceptron->layer;

	return layer->activation.derivative(perceptron->z);
}

static int backward_propagate_layer(struct layer *layer, struct sample *sample) {
	if(layer == NULL || layer->network == NULL) return -1;

	struct network *network = layer->network;

	for(int i = 0; i < layer->n; i++) {
		struct perceptron *perceptron = &layer->perceptrons[i];

		for(int j = 0; j < perceptron->n; j++) {
			struct signal *signal = &perceptron->signals[i];

			signal->weight -= network->learning_rate * cost_gradient(perceptron, signal, sample);
		}
	}

	return 0;
}

int backward_propagate(struct network *network, struct sample *sample) {
	if(network == NULL || sample == NULL) return -1;

	struct layer *root = network->output;

	for(;;) { 
		if(root == NULL) break;

		int ret = backward_propagate_layer(root, sample);
		if(ret == -1) return -1;

		root = root->parent;
	}

	return 0;
}
