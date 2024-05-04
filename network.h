#ifndef NETWORK_H_
#define NETWORK_H_

#include <activation.h>

#include <stdlib.h>

struct perceptron;

struct signal {
	struct perceptron *emitter;
	float weight;
};

struct perceptron {
	float x;
	float y;

	int n;

	struct layer *layer;
	struct signal *signals;
};

struct layer {
	int n; 
	int bias;

	struct layer *child; 
	struct layer *parent;

	struct activation activation;

	struct perceptron perceptrons[];
};

struct network {
	struct training_set *training_set;

	struct layer *input; 
	struct layer *hidden;
	struct layer *output; 
};

#define CREATE_HIDDEN_LAYER(N, BIAS, ACTIVATION) ({ \
	struct layer *_layer = calloc(sizeof(struct layer) + \
			sizeof(struct perceptron) * (N), 1); \
	_layer->n = N; \
	_layer->bias = BIAS; \
	_layer->activation = ACTIVATION; \
	_layer; \
})

int initialise_layers(struct layer *root, ...);
int display_network(struct network *network);

struct layer *create_input_layer(struct layer *joint, int n);
struct layer *create_output_layer(struct layer *joint, struct activation activation, int n);

#endif
