#include <mnist.h>

#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>

#include <sys/mman.h>
#include <sys/stat.h>

int mnist_get_sample(void *private, struct sample *sample, int n) {
	struct mnist_training_data *data = private;

	struct mnist_image image;
	struct mnist_label label;

	int ret = mnist_get_image(data->image_set, &image, n);
	if(ret == -1) return -1; 

	ret = mnist_get_label(data->label_set, &label, n);
	if(ret == -1) return -1;

	sample->length = image.size;
	sample->data = image.data;
	sample->expected = label.expected;

	return 0;
}

int mnist_get_image(struct mnist_image_set *image_set, struct mnist_image *image, uint32_t n) {
	if(image_set == NULL || image == NULL ||
			image_set->hdr.image_cnt < n) return -1;

	image->row_cnt = image_set->hdr.row_cnt;
	image->column_cnt = image_set->hdr.column_cnt;
	image->size = image_set->hdr.column_cnt * image_set->hdr.row_cnt;

	image->data = image_set->data + sizeof(struct mnist_image_hdr) + n * image->size;

	return 0;
}

int mnist_get_label(struct mnist_label_set *label_set, struct mnist_label *label, uint32_t n) {
	if(label_set == NULL || label == NULL ||
			label_set->hdr.item_cnt < n) return -1;

	label->expected = *(uint8_t*)(label_set->data + sizeof(struct mnist_label_hdr) + n);

	return 0;
}

int mnist_training_init(struct training_set *training_set) {
	if(training_set == NULL) return -1;

	struct mnist_training_data *mnist_training_data = malloc(sizeof(struct mnist_training_data));

	int fd_images = open(MNIST_IMAGES_PATH, O_RDONLY);
	if(fd_images == -1) {
		printf("antwerp: can not open \"%s\"\n", MNIST_IMAGES_PATH);
		return -1;
	}

	int fd_labels = open(MNIST_LABELS_PATH, O_RDONLY);
	if(fd_labels == -1) {
		printf("antwerp: can not open \"%s\"\n", MNIST_LABELS_PATH);
		return -1;
	}

	struct mnist_image_set *image_set = malloc(sizeof(struct mnist_image_set));
	image_set->fd = fd_images;

	int ret = stat(MNIST_IMAGES_PATH, &image_set->stat);
	if(ret == -1) return -1;

	image_set->data = mmap(NULL, image_set->stat.st_size, PROT_READ, MAP_PRIVATE,
			image_set->fd, 0);
	if(image_set->data == (void*)-1) return -1;

	struct mnist_label_set *label_set = malloc(sizeof(struct mnist_label_set));
	label_set->fd = fd_labels;

	ret = stat(MNIST_LABELS_PATH, &label_set->stat);
	if(ret == -1) return -1;

	label_set->data = mmap(NULL, label_set->stat.st_size, PROT_READ, MAP_PRIVATE,
			label_set->fd, 0);
	if(label_set->data == (void*)-1) return -1;

	image_set->hdr = (struct mnist_image_hdr) {
		.magic_number = __builtin_bswap32(*(((uint32_t*)image_set->data) + 0)),
		.image_cnt = __builtin_bswap32(*(((uint32_t*)image_set->data) + 1)),
		.row_cnt = __builtin_bswap32(*(((uint32_t*)image_set->data) + 2)),
		.column_cnt = __builtin_bswap32(*(((uint32_t*)image_set->data) + 3))
	};

	if(image_set->hdr.magic_number != MNIST_IMAGE_SIGNATURE) return -1;

	label_set->hdr = (struct mnist_label_hdr) {
		.magic_number = __builtin_bswap32(*(((uint32_t*)label_set->data) + 0)),
		.item_cnt = __builtin_bswap32(*(((uint32_t*)label_set->data) + 1)),
	};

	if(label_set->hdr.magic_number != MNIST_LABEL_SIGNATURE) return -1;

	printf("antwerp: mnist: column: %d\n", image_set->hdr.column_cnt);
	printf("antwerp: mnist: row: %d\n", image_set->hdr.row_cnt);

	mnist_training_data->image_set = image_set;
	mnist_training_data->label_set = label_set;

	training_set->private = mnist_training_data;
	training_set->get_sample = mnist_get_sample;

	training_set->output_nodes = 10;
	training_set->input_nodes = image_set->hdr.row_cnt * image_set->hdr.column_cnt;
	training_set->hidden_nodes = 400;

	return 0;
}
