#ifndef ANTWERP_H_
#define ANTWERP_H_

#include <math.h>

#define SIGMOID ({ \
	(struct function) {\
		.function = sigmoid, \
		.derivative = sigmoid_derivative \
	}; \
})

#define RELU ({ \
	(struct function) { \
		.function = relu, \
		.derivative = relu_derivative \
	}; \
})

struct function {
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

static inline double cost_mse(double y, double expected) {
	return (expected - y) * (expected - y);
}

static inline double cost_mse_derivative(double y, double expected) {
	return 2 * (expected - y);
}

#endif
