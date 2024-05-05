#ifndef TRAINING_H_
#define TRAINING_H_

#include <network.h>

#include <math.h> 

struct sample {
	int length;
	void *data;

	int expected;
};

struct training_set {
	struct network *network;

	int input_nodes;
	int hidden_nodes;
	int output_nodes;

	double (*cost)(struct sample*, double);

	void *private;
	int (*get_sample)(void*, struct sample*, int);
};

int train(struct network *network);

#endif
