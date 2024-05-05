#include <network.h>

#include <stdarg.h>
#include <stdio.h>

static int initialise_layer(struct network*, struct layer*);
static int sew_signals(struct layer*);
static int display_layer(struct layer*);

#define RANDOM_WEIGHT (2.0 * ((double)rand() / RAND_MAX) - 1.0) * 0.2;

static int initialise_layer(struct network *network, struct layer *layer) {
	if(network == NULL || layer == NULL) return -1;

	layer->network = network;

	for(int i = 0; i < layer->n; i++) {
		struct perceptron *perceptron = &layer->perceptrons[i];

		perceptron->layer = layer;
		perceptron->n = layer->n;
		perceptron->signals = calloc(sizeof(struct signal) *
				perceptron->n, 1);

		for(int j = 0; j < perceptron->n; j++) {
			perceptron->signals[j].weight = RANDOM_WEIGHT;
		}
	}

	return 0;
}

// Potentially something worth doing in future, define an interface
// that allows you to sew more sophisticated connections
static int sew_signals(struct layer *layer) {
	if(layer == NULL || layer->parent == NULL) return -1;

	struct layer *parent = layer->parent;

	for(int i = 0; i < layer->n; i++) {
		struct perceptron *perceptron = &layer->perceptrons[i];

		for(int j = 0; j < perceptron->n; j++) {
			perceptron->signals[j].emitter = &parent->perceptrons[j];
		}
	}

	return 0;
}

// connect all nodes in the input layer to the perceptrons in the joint layer
struct layer *create_input_layer(struct network *network, struct layer *joint, int n) {
	if(network == NULL || joint == NULL) return NULL;

	struct layer *input = calloc(sizeof(struct layer) + sizeof(struct perceptron) * n, 1);
	input->n = n;

	for(int i = 0; i < joint->n; i++) {
		struct perceptron *perceptron = &joint->perceptrons[i];

		perceptron->n = input->n;
		perceptron->signals = calloc(sizeof(struct signal) * perceptron->n, 1);
		perceptron->layer = input;

		for(int j = 0; j < input->n; j++) {
			perceptron->signals[j].emitter = &input->perceptrons[j];
			perceptron->signals[j].weight = RANDOM_WEIGHT;
		}
	}

	input->network = network;
	network->input = input;

	return input;
}

// connect all perceptrons the joint layer to the nodes in the output layer
struct layer *create_output_layer(struct network *network, struct layer *joint,
		struct function activation, int n) {
	if(network == NULL || joint == NULL) return NULL;

	struct layer *output = calloc(sizeof(struct layer) + sizeof(struct perceptron) * n, 1);

	output->n = n;
	output->activation = activation;

	for(int i = 0; i < output->n; i++) {
		struct perceptron *perceptron = &output->perceptrons[i];

		perceptron->n = joint->n;
		perceptron->signals = calloc(sizeof(struct signal) * perceptron->n, 1);
		perceptron->layer = output;

		for(int j = 0; j < joint->n; j++) {
			perceptron->signals[j].emitter = &joint->perceptrons[j];
			perceptron->signals[j].weight = RANDOM_WEIGHT;
		}
	}

	output->network = network;
	network->output = output;

	return output;
}

int initialise_layers(struct network *network, struct layer *root, ...) {
	if(network == NULL) return -1;

	va_list args;
	va_start(args, root);

	if(root == NULL) {
		root = va_arg(args, struct layer*);
		if(root == NULL) return -1;
	}

	int ret = initialise_layer(network, root);
	if(ret == -1) return -1;

	for(;;) { 
		struct layer *layer = va_arg(args, struct layer*);
		if(layer == NULL) break;

		ret = initialise_layer(network, layer);
		if(ret == -1) continue;

		root->child = layer;
		layer->parent = root;
		root = layer;
	}

	for(;;) {
		if(root == NULL) break;

		network->hidden = root;

		sew_signals(root);

		root = root->parent;
	}

	va_end(args);

	return 0;
}

struct perceptron *layer_output(struct layer *layer) {
	if(layer == NULL) return NULL;

	struct perceptron *perceptron = NULL;
	for(int i = 0; i < layer->n; i++) {
		if(perceptron == NULL || (layer->perceptrons[i].a > perceptron->a)) {
			perceptron = &layer->perceptrons[i];
		}
	}

	return perceptron;
}

static int display_layer(struct layer *layer) {
	for(int i = 0; layer && i < layer->n; i++) {
		struct perceptron *perceptron = &layer->perceptrons[i];

		printf("%f ", perceptron->a);
	}

	printf("\n");

	return 0;
}

int display_network(struct network *network, int flags) {
	if(network == NULL) return -1;

	int line = 0;

	if((flags & DISPLAY_HIDE_INPUT) == 0) {
		printf("antwerp: input layer:\n\t%d: ", line++);
		display_layer(network->input);
	} else if((flags & DISPLAY_HIDE_HIDDEN) == 0) {
		printf("antwerp: hidden layer(s):\n");

		struct layer *root = network->hidden;

		for(;; line++) {
			if(root == NULL) break;
			printf("\t%d: ", line);
			display_layer(root);
			root = root->child;
		}
	} else if((flags & DISPLAY_HIDE_OUTPUT) == 0) {
		printf("antwerp: ouput layer:\n\t%d: ", line);
		display_layer(network->output);
	}

	return 0;
}
