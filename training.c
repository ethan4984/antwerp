#include <training.h>
#include <propagation.h>

#include <stdio.h>

int train(struct network *network) {
	if(network == NULL || network->training_set == NULL) return -1;

	struct training_set *training_set = network->training_set;

	for(int n = 0;; n++) {
		struct sample sample;

		int ret = training_set->get_sample(training_set->private, &sample, n);
		if(ret == -1) break;

		for(int i = 0; i < sample.length && i < network->input->n; i++) {
			network->input->perceptrons[i].y = fabs(*(char*)(sample.data + i) / (double)255);
		}

		ret = forward_propagate(network);
		if(ret == -1) {
			printf("antwerp: failure during forward propagation on sample n=%d\n", n);
			return -1;
		}

		printf("-----------------\nantwerp: expected output: y=%d\n", sample.expected);
		display_network(network, DISPLAY_HIDE_INPUT | DISPLAY_HIDE_HIDDEN);
	}

	return 0;
}
