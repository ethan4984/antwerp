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
	double (*function)(double);
	double (*derivative)(double);
};

static inline double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

static inline double sigmoid_derivative(double x) {
	return exp(-x) / ((1 + exp(-x)) * (1 + exp(-x)));
}

static inline double relu(double x) {
	if(x) return x;
	else return 0.0;
}

static inline double relu_derivative(double x) {
	if(x) return 1.0;
	else return 0.0;
}

#endif
