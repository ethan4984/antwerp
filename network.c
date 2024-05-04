#include <network.h>

#include <stdarg.h>
#include <stdio.h>

static int initialise_layer(struct layer*);
static int sew_signals(struct layer*);
static int display_layer(struct layer*);

static int initialise_layer(struct layer *layer) {
	if(layer == NULL) return -1;

	for(int i = 0; i < layer->n; i++) {
		struct perceptron *perceptron = &layer->perceptrons[i];

		perceptron->layer = layer;
		perceptron->n = layer->n;
		perceptron->signals = calloc(sizeof(struct signal) *
				perceptron->n, 1);

		for(int j = 0; j < perceptron->n; j++) {
			perceptron->signals[j].weight = (float)rand() / RAND_MAX;
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
struct layer *create_input_layer(struct layer *joint, int n) {
	if(joint == NULL) return NULL;

	struct layer *input = calloc(sizeof(struct layer) + sizeof(struct perceptron) * n, 1);
	input->n = n;

	for(int i = 0; i < joint->n; i++) {
		struct perceptron *perceptron = &joint->perceptrons[i];

		perceptron->n = input->n;
		perceptron->signals = calloc(sizeof(struct signal) * perceptron->n, 1);

		for(int j = 0; j < input->n; j++) {
			perceptron->signals[j].emitter = &input->perceptrons[j];
			perceptron->signals[j].weight = (float)rand() / RAND_MAX;
		}
	}

	return input;
}

// connect all perceptrons the joint layer to the nodes in the output layer
struct layer *create_output_layer(struct layer *joint, struct activation activation, int n) {
	if(joint == NULL) return NULL;

	struct layer *output = calloc(sizeof(struct layer) + sizeof(struct perceptron) * n, 1);

	output->n = n;
	output->activation = activation;

	for(int i = 0; i < output->n; i++) {
		struct perceptron *perceptron = &output->perceptrons[i];

		perceptron->n = joint->n;
		perceptron->signals = calloc(sizeof(struct signal) * perceptron->n, 1);

		for(int j = 0; j < joint->n; j++) {
			perceptron->signals[j].emitter = &joint->perceptrons[j];
			perceptron->signals[j].weight = (float)rand() / RAND_MAX;
		}
	}

	return output;
}

int initialise_layers(struct layer *root, ...) {
	va_list args;
	va_start(args, root);

	if(root == NULL) {
		root = va_arg(args, struct layer*);
		if(root == NULL) return -1;
	}

	int ret = initialise_layer(root);
	if(ret == -1) return -1;

	for(;;) { 
		struct layer *layer = va_arg(args, struct layer*);
		if(layer == NULL) break;

		ret = initialise_layer(layer);
		if(ret == -1) continue;

		root->child = layer;
		layer->parent = root;
		root = layer;
	}

	for(;;) {
		if(root == NULL) break;
		sew_signals(root);
		root = root->parent;
	}

	va_end(args);

	return 0;
}

static int display_layer(struct layer *layer) {
	for(int i = 0; layer && i < layer->n; i++) {
		struct perceptron *perceptron = &layer->perceptrons[i];

		printf("%f ", perceptron->y);
	}

	printf("\n");

	return 0;
}

int display_network(struct network *network) {
	if(network == NULL) return -1;

	printf("antwerp: input layer:\n\t1: ");

	display_layer(network->input);

	printf("antwerp: hidden layer(s):\n");

	struct layer *root = network->hidden;

	int i = 2;
	for(;; i++) {
		if(root == NULL) break;
		printf("\t%d: ", i);
		display_layer(root);
		root = root->child;
	}

	printf("antwerp: ouput layer:\n\t%d: ", i);

	display_layer(network->output);

	return 0;
}
