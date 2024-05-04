#ifndef MNIST_H_
#define MNIST_H_

#include <training.h>

#include <sys/stat.h>

#include <stdint.h>

#define MNIST_IMAGES_PATH "data/train-images-idx3-ubyte"
#define MNIST_LABELS_PATH "data/train-labels-idx1-ubyte"

#define MNIST_IMAGE_SIGNATURE 0x803
#define MNIST_LABEL_SIGNATURE 0x801

struct mnist_image_hdr {
	uint32_t magic_number;
	uint32_t image_cnt;
	uint32_t row_cnt;
	uint32_t column_cnt;
} __attribute__((packed));

struct mnist_label_hdr {
	uint32_t magic_number;
	uint32_t item_cnt;
} __attribute__((packed));

struct mnist_image_set {
	int fd;
	struct stat stat;

	struct mnist_image_hdr hdr;
	void *data;
};

struct mnist_image {
	void *data;

	uint32_t size;
	uint32_t row_cnt;
	uint32_t column_cnt;
};

struct mnist_label_set {
	int fd;
	struct stat stat;

	struct mnist_label_hdr hdr;
	void *data;
};

struct mnist_label {
	uint8_t expected;
};

struct mnist_training_data {
	struct mnist_image_set *image_set; 
	struct mnist_label_set *label_set;
};

int mnist_training_init(struct training_set *training_set);
int mnist_get_image(struct mnist_image_set *image_set, struct mnist_image *image, uint32_t n);
int mnist_get_label(struct mnist_label_set *label_set, struct mnist_label *label, uint32_t n);

#endif
