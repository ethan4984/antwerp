#ifndef TRAINING_H_
#define TRAINING_H_

#include <network.h>

#include <math.h> 

struct sample;

struct training_set {
	struct network *network;

	int input_nodes;
	int output_nodes;

	float (*cost)(struct sample*, void*, float);

	int sample_cnt;
	struct sample *samples;
};

struct sample {
	int length;
	void *data;
	int expected;
};

static inline float cost_mse(struct sample *sample, void*, float y) {
	if(sample == NULL) return NAN;
	return (sample->expected - y) * (sample->expected - y);
}

#endif
