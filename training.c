#include <training.h>
#include <propagation.h>

#include <stdio.h>
#include <stdint.h>

int train(struct network *network) {
	if(network == NULL || network->training_set == NULL) return -1;

	struct training_set *training_set = network->training_set;

	for(int n = 0;; n++) {
		struct sample sample;

		int ret = training_set->get_sample(training_set->private, &sample, n);
		if(ret == -1) break;

		for(int i = 0; i < sample.length && i < network->input->n; i++) {
			network->input->perceptrons[i].a = fabs(*(char*)(sample.data + i) / (double)255);
		}

		ret = forward_propagate(network);
		if(ret == -1) {
			printf("antwerp: failure during forward propagation on sample n=%d\n", n);
			return -1;
		}

		struct perceptron *result = layer_output(network->output);
		if(result == NULL) return -1;

		printf("-----------------\nantwerp: expected output: %d\n", sample.expected);
		display_network(network, DISPLAY_HIDE_INPUT | DISPLAY_HIDE_HIDDEN);
		printf("antwerp: output: %ld\n", ((uintptr_t)result - (uintptr_t)network->output->perceptrons)
				/ sizeof(struct perceptron));

		ret = backward_propagate(network, &sample);
		if(ret == -1) {
			printf("antwerp: failure during back propagation on sample n=%d\n", n);
			return -1;
		}
	}

	return 0;
}
