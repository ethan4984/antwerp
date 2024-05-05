#ifndef PROPAGATION_H_
#define PROPAGATION_H_

#include <network.h>
#include <training.h>

int forward_propagate(struct network*);
int backward_propagate(struct network*, struct sample*);

#endif
