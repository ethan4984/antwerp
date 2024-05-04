PROGRAM := antwerp

CC ?= cc
LD ?= $(CC)

CFLAGS  ?= -Wall -Wextra -O0 -pipe -c
LDFLAGS ?= -lm

INTERNALCFLAGS := \
	-I. \
	-std=gnu11

CFILES := $(shell find ./ -type f -name '*.c')
OBJ := $(CFILES:.c=.o)
HEADER_DEPS := $(CFILES:.c=.d)

.PHONY: all
all: $(PROGRAM)

$(PROGRAM): $(BINS) $(OBJ)
	$(CC) $(OBJ) $(LDFLAGS) $(INTERNALLDFLAGS) -o $@

-include $(HEADER_DEPS)

%.o: %.c
	$(CC) $(CFLAGS) $(INTERNALCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -rf $(PROGRAM) $(OBJ) $(HEADER_DEPS)

.PHONY: run
run: $(PROGRAM)
	@./$(PROGRAM)
