#include <propagation.h>

#include <stddef.h>
#include <stdio.h>

int forward_propagate(struct network *network) {
	if(network == NULL) return -1;

	for(int i = 1; i < network->layers; i++) {
		for(int k = 0; k < network->n[i]; k++) {
			network->z[i][k] = network->biases[i];

			for(int j = 0; j < network->n[i - 1]; j++) {
				network->z[i][k] += network->a[i - 1][j] * network->weights[i][k][j];
			}

			network->a[i][k] = network->activation.function(network->z[i][k]);
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

	for(int k = 0; k < network->n[layer]; k++) {
		double dadz = network->activation.derivative(network->z[layer][k]);
		double dcda = 2 * (network->a[layer][k] - network->expected[k]);
		double dcdb = dadz * dcda;

		network->dcdb[layer][k] = dcdb;

		for(int j = 0; j < network->n[layer - 1]; j++) {
			double dzdw = network->a[layer - 1][j]; 
			double dcdw = dzdw * dadz * dcda;

			network->dcdw[layer][k][j] = dcdw;

#ifdef ANTWERP_DEBUG
			printf("antwerp: dcdw on L=%d j=%d k=%d: %f\n", layer, j, k, dcdw);
			printf("\tdzdw: %f\n", dzdw);
			printf("\tdadz: %f\n", dadz);
			printf("\tdcda: %f\n", dcda);
#endif
		}
	}

	for(; layer-- > 1;) {
		for(int k = 0; k < network->n[layer]; k++) {
			double dadz = network->activation.derivative(network->z[layer][k]);
			double dcda = activation_sum(network, layer + 1, k);

			double dcdb = dadz * dcda;

			network->dcdb[layer][k] = dcdb;

			for(int j = 0; j < network->n[layer - 1]; j++) {
				double dzdw = network->a[layer - 1][j];
				double dcdw = dzdw * dadz * dcda;

				network->dcdw[layer][k][j] = dcdw;

#ifdef ANTWERP_DEBUG
				printf("antwerp: dcdw on L=%d j=%d k=%d: %f\n", layer, j, i, dcdw);
				printf("\tdzdw: %f\n", dzdw);
				printf("\tdadz: %f\n", dadz);
				printf("\tdcda: %f\n", dcda);
#endif
			}
		}
	}

	for(int i = network->layers; i-- > 1;) {
		for(int k = 0; k < network->n[i]; k++) {
			network->biases[i] -= network->learning_rate * network->dcdb[i][k];

			for(int j = 0; j < network->n[i - 1]; j++) {
				network->weights[i][k][j] -= network->learning_rate * network->dcdw[i][k][j];
			}
		}
	}

	return 0;
}
