#ifndef NETWORK_H_
#define NETWORK_H_

#include <training.h>
#include <antwerp.h>

struct network {
	struct training_set *training_set;
	double learning_rate;

	struct function activation;
	int batch_size;
	int layers;

	double ***weights;
	double ***dcdw;

	double **dcdb;
	double **a;
	double **z;

	double *expected;
	double *biases;
	int *n;
};

#define DISPLAY_WEIGHTS (1 << 0)
#define DISPLAY_ACTIVATIONS (1 << 1)
#define DISPLAY_INPUT (1 << 2)
#define DISPLAY_OUTPUT (1 << 3)
#define DISPLAY_HIDDEN (1 << 4)

int network_init(struct network*, int, struct function, ...);
int network_display(struct network*, int flags);

#endif
