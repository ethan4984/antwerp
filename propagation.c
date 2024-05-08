#include <propagation.h>

#include <stddef.h>
#include <stdio.h>

int forward_propagate(struct network *network) {
	if(network == NULL) return -1;

	for(int i = 1; i < network->layers; i++) {
		for(int j = 0; j < network->n[i]; j++) {
			network->z[i][j] = network->biases[i];

			for(int k = 0; k < network->n[i - 1]; k++) {
				network->z[i][j] += network->a[i - 1][k] * network->weights[i][j][k];
			}

			network->a[i][j] = network->activation.function(network->z[i][j]);
		}
	}

	return 0;
}

static double activation_sum(struct network *network, int layer, int k) {
	double sum = 0.0;

	for(int j = 0; j < network->n[layer]; j++) {
		double dcda = (layer == network->layers - 1) ? 2 * (network->a[layer][j] - network->expected[j]) :
			activation_sum(network, layer + 1, j);

		sum += network->weights[layer][j][k] *
			network->activation.derivative(network->z[layer][j]) * dcda;
	}

	return sum;
}

int backward_propagate(struct network *network) {
	if(network == NULL) return -1;

	int layer = network->layers - 1;

	for(int i = 0; i < network->n[layer]; i++) {
		for(int j = 0; j < network->n[layer - 1]; j++) {
			double dzdw = network->a[layer - 1][j]; 
			double dadz = network->activation.derivative(network->z[layer][i]);
			double dcda = 2 * (network->a[layer][i] - network->expected[i]);

			double dcdw = dzdw * dadz * dcda;

			network->dcda[layer][i][j] = dcda;
			network->dcdw[layer][i][j] = dcdw;

#ifdef ANTWERP_DEBUG
			printf("antwerp: dcdw on L=%d k=%d j=%d: %f\n", layer, j, i, dcdw);
			printf("\tdzdw: %f\n", dzdw);
			printf("\tdadz: %f\n", dadz);
			printf("\tdcda: %f\n", dcda);
#endif
		}
	}

	for(; layer-- > 1;) {
		for(int i = 0; i < network->n[layer]; i++) {
			for(int j = 0; j < network->n[layer - 1]; j++) {
				double dzdw = network->a[layer - 1][j];
				double dadz = network->activation.derivative(network->z[layer][i]);
				double dcda = activation_sum(network, layer + 1, i);

				double dcdw = dzdw * dadz * dcda;

				network->dcdw[layer][i][j] = dcdw;
				network->dcda[layer][i][j] = dcda;

#ifdef ANTWERP_DEBUG
				printf("antwerp: dcdw on L=%d k=%d j=%d: %f\n", layer, j, i, dcdw);
				printf("\tdzdw: %f\n", dzdw);
				printf("\tdadz: %f\n", dadz);
				printf("\tdcda: %f\n", dcda);
#endif
			}
		}
	}

	for(int i = network->layers; i-- > 1;) {
		for(int j = 0; j < network->n[i]; j++) {
			for(int k = 0; k < network->n[i - 1]; k++) {
				network->weights[i][j][k] -= network->learning_rate * network->dcdw[i][j][k];
			}
		}
	}

	return 0;
}
