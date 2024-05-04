#ifndef ACTIVATION_H_
#define ACTIVATION_H_

#include <math.h>

#define SIGMOID ({ \
	struct activation _activation = { \
		.function = sigmoid, \
		.derivative = sigmoid_derivative \
	}; \
	_activation; \
})

#define RELU ({ \
	(struct activation) { \
		.function = relu, \
		.derivative = relu_derivative \
	}; \
})

struct activation {
	float (*function)(float);
	float (*derivative)(float);
};

static inline float sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x));
}

static inline float sigmoid_derivative(float x) {
	return exp(-x) / ((1 + exp(-x)) * (1 + exp(-x)));
}

static inline float relu(float x) {
	if(x) return x;
	else return 0.0;
}

static inline float relu_derivative(float x) {
	if(x) return 1.0;
	else return 0.0;
}

#endif
