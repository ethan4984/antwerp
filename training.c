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
			network->input->perceptrons[i].y = *(char*)(sample.data + i);
		}

		ret = forward_propagate(network);
		if(ret == -1) {
			printf("antwerp: failure during forward propagation on sample n=%d\n", n);
			return -1;
		}
	}

	return 0;
}
