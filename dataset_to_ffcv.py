import numpy as np
from ffcv import DatasetWriter
from ffcv.fields import NDArrayField

import dnnlib
from training.dataset import Dataset


def main():
    training_set_kwargs = {
        "class_name": "training.dataset.ImageFolderDataset",
        "path": "datasets/cifar/cifar_10",
        # "path": "datasets/cifar/cifar_10.zip",
        "use_labels": True,
        "max_size": 50000,
        "xflip": True,
        "yflip": False,
        "resolution": 32,
        "random_seed": 0
    }
    #
    # training_set_kwargs= {
    #     "class_name": "training.dataset.ImageFolderDataset",
    #     "path": "datasets/pixel-256/pixel-256.zip",
    #     "use_labels": False,
    #     "max_size": 2144,
    #     "xflip": True,
    #     "yflip": False,
    #     "resolution": 256,
    #     "random_seed": 0
    # }

    with open('datasets/cifar/cifar_10.zip', 'rb') as MyZip:
        print(MyZip.read(4))

    N, d = (100, 6)

    training_set: Dataset = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    test_label = training_set.get_label(1)

    write_path = 'datasets/cifar/cifar_10.beton'
    # write_path = 'datasets/cifar/pixel-256.beton'

    writer = DatasetWriter(write_path, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': NDArrayField(dtype=np.dtype(np.uint8), shape=training_set.image_shape),
        # 'image': RGBImageField(),
        'label': NDArrayField(dtype=test_label.dtype, shape=test_label.shape)
        # 'label': IntField()
    })
    writer.from_indexed_dataset(training_set)


if __name__ == '__main__':
    main()
