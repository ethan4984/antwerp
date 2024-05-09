#include <network.h>

#include <stdlib.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdio.h>

#define RANDOM_WEIGHT (2.0 * ((double)rand() / RAND_MAX) - 1.0) * 0.1;

static void display_layer_activations(struct network *network, int layer);
static void display_layer_weights(struct network *network, int layer);

static int network_init_weights(struct network *network, int layer, int n) {
	if(network == NULL || network->weights == NULL) return -1;

	for(int i = 0; i < network->n[layer]; i++) {
		for(int j = 0; j < n; j++) {
			network->weights[layer][i][j] = RANDOM_WEIGHT;
		}
	}

	return 0;
}

int network_init(struct network *network, int layers, struct function activation, ...) {
	if(network == NULL || layers == 0) return -1;

	va_list args;
	va_start(args, activation);

	network->activation = activation;

	network->weights = malloc(sizeof(double**) * layers);
	network->dcdw = malloc(sizeof(double**) * layers);
	network->dcdb = malloc(sizeof(double*) * layers);
	network->a = malloc(sizeof(double*) * layers);
	network->z = malloc(sizeof(double*) * layers);
	network->n = malloc(sizeof(int) * layers);
	network->biases = malloc(sizeof(double) * layers);

	// input layers;
	
	int n = va_arg(args, int);
	int bias = va_arg(args, int);

	network->n[0] = n;
	network->biases[0] = bias;
	network->a[0] = malloc(sizeof(double) * n);
	network->weights[0] = NULL;
	network->dcdw[0] = NULL;
	network->dcdw[0] = NULL;
	network->z[0] = NULL;

	network->layers = 1;

	// hidden and output layers
	for(; network->layers < layers;) {
		int n = va_arg(args, int);
		int bias = va_arg(args, int);

		network->biases[network->layers] = bias;
		network->n[network->layers] = n;

		int total_weights = network->n[network->layers - 1];

		network->weights[network->layers] = malloc(sizeof(double*) * n);
		network->dcdw[network->layers] = malloc(sizeof(double*) * n);
		for(int i = 0; i < n; i++) {
			network->weights[network->layers][i] = malloc(sizeof(double) * total_weights);
			network->dcdw[network->layers][i] = malloc(sizeof(double) * total_weights);
		}

		network->a[network->layers] = malloc(sizeof(double) * n);
		network->z[network->layers] = malloc(sizeof(double) * n);
		network->dcdb[network->layers] = malloc(sizeof(double) * n);

		int ret = network_init_weights(network, network->layers, total_weights);
		if(ret == -1) return -1;

		network->layers++;
	}

	network->expected = malloc(sizeof(double) * network->n[network->layers - 1]);

	va_end(args);

	return 0;
}

static void display_layer_activations(struct network *network, int layer) {
	for(int i = 0; i < network->n[layer]; i++) {
		printf("%f ", network->a[layer][i]);
	}
	printf("\n");
}

static void display_layer_weights(struct network *network, int layer) {
	printf("1: ");
	for(int i = 0; i < network->n[layer];) {
		for(int j = 0; j < network->n[layer - 1]; j++) {
			printf("%f ", network->weights[layer][i][j]);
		}
		if(++i < network->n[layer]) printf("\n\t\t%d: ", i + 1);
	}
	printf("\n");
}

int network_display(struct network *network, int flags) {
	if(network == NULL) return -1;

	printf("-----------------------------------\n");

	if(flags & DISPLAY_INPUT) { 
		if(flags & DISPLAY_ACTIVATIONS) {
			printf("antwerp: input layer\n");
			printf("\tactivations:\n\t\t");
			display_layer_activations(network, 0);
		}
	}

	if(flags & DISPLAY_HIDDEN)  {
		for(int layer = 1; layer < network->layers - 1; layer++) {
			printf("antwerp: hidden layers %d\n", layer);
			if(flags & DISPLAY_WEIGHTS) {
				printf("\tweights:\n\t\t");
				display_layer_weights(network, layer);
			} if(flags & DISPLAY_ACTIVATIONS) {
				printf("\tactivations:\n\t\t");
				display_layer_activations(network, layer);
			}
		}
	}

	if(flags & DISPLAY_OUTPUT)  {
		printf("antwerp: output layer\n");
		if(flags & DISPLAY_WEIGHTS) {
			printf("\tweights:\n\t\t");
			display_layer_weights(network, network->layers - 1);
		} if(flags & DISPLAY_ACTIVATIONS) {
			printf("\tactivations:\n\t\t");
			display_layer_activations(network, network->layers - 1);
		}
	}

	printf("-----------------------------------\n");

	return 0;
}
