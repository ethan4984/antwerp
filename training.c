#include <training.h>

#include <stdio.h>
#include <stdint.h>

int train(struct network *network) {
	if(network == NULL || network->training_set == NULL) return -1;

	struct training_set *training_set = network->training_set;

	int a = 0;

	for(int n = 0;; n++) {
		struct sample sample;

		int ret = training_set->get_sample(training_set->private, &sample, n);
		if(ret == -1) break;

		for(int i = 0; i < sample.length && i < network->n[0]; i++) {
			network->a[0][i] = ((double)(*(uint8_t*)(sample.data + i))) / (double)255;
		}

		ret = forward_propagate(network);
		if(ret == -1) {
			printf("antwerp: failure during forward propagation on sample n=%d\n", n);
			return -1;
		}

		int result;
		double tmp = -1.0;
		for(int i = 0; i < network->n[network->layers - 1]; i++) {
			if(network->a[network->layers - 1][i] > tmp) {
				tmp = network->a[network->layers - 1][i];
				result = i;
			}
		}

		double cost = 0.0;
		for(int i = 0; i < network->n[network->layers - 1]; i++) {
			cost += (network->a[network->layers - 1][i] - network->expected[i]) *
				(network->a[network->layers - 1][i] - network->expected[i]);
		}

		printf("-----------------\nantwerp: expected output: %d\n", sample.expected);
		printf("antwerp: output layer:\n\t0: ");
		network_display_layer(network, 2);
		printf("antwerp: result: %d with cost %f\n", result, cost);

		for(int i = 0; i < network->n[network->layers - 1]; i++) {
			network->expected[i] = (i == sample.expected) ? 1.0 : 0.0;
		}

		ret = backward_propagate(network, &sample);
		if(ret == -1) {
			printf("antwerp: failure during backwards propagation on sample n=%d\n", n);
			return -1;
		}
	}

	return 0;
}
