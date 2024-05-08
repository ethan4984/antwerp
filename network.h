#ifndef NETWORK_H_
#define NETWORK_H_

#include <training.h>
#include <antwerp.h>

struct network {
	struct training_set *training_set;
	double learning_rate;

	struct function activation;
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

int network_init(struct network*, int, struct function, ...);
int network_display(struct network*);
void network_display_layer(struct network*, int);

#endif
