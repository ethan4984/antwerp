#ifndef NETWORK_H_
#define NETWORK_H_

#include <antwerp.h>

#include <stdlib.h>

struct perceptron;

struct signal {
	struct perceptron *emitter;
	double weight;
};

struct perceptron {
	double z;
	double a;

	int n;

	struct layer *layer;
	struct signal *signals;
};

struct layer {
	int n; 
	int bias;

	struct network *network;
	struct layer *child; 
	struct layer *parent;

	struct function activation;

	struct perceptron perceptrons[];
};

struct network {
	struct training_set *training_set;

	double learning_rate;

	struct layer *input; 
	struct layer *hidden;
	struct layer *output; 
};

#define CREATE_HIDDEN_LAYER(NETWORK, N, BIAS, ACTIVATION) ({ \
	struct layer *_layer = calloc(sizeof(struct layer) + \
			sizeof(struct perceptron) * (N), 1); \
	_layer->n = N; \
	_layer->bias = BIAS; \
	_layer->activation = ACTIVATION; \
	_layer->network = NETWORK; \
	_layer; \
})

#define DISPLAY_HIDE_INPUT (1 << 0)
#define DISPLAY_HIDE_OUTPUT (1 << 1)
#define DISPLAY_HIDE_HIDDEN (1 << 2)

int initialise_layers(struct network*, struct layer*, ...);
int display_network(struct network*, int);

struct layer *create_input_layer(struct network*, struct layer*, int);
struct layer *create_output_layer(struct network*, struct layer*, struct function, int);

struct perceptron *layer_output(struct layer*);

#endif
