import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF

from src.const import root_dir, original_dir_name, interpolated_dir_name

patch_width = 32
patch_height = 32


def list_all_images():
    path = Path(root_dir, original_dir_name)
    topics = []
    for folder in path.iterdir():
        images = []
        for file in path.joinpath(folder.name).iterdir():
            images.append(f'{folder.name}/{file.name}')
        topics.append(images)

    import json
    with open('../dataset/all.json', 'w') as f:
        f.write(json.dumps(topics))


def train_test_split(training_r=.7, test_r=.2):
    assert training_r + test_r <= 1, 'Sum of ratios should be less than 1!'

    with open('../dataset/all.json', 'r') as f:
        image_files = json.load(f)

    x_train = []
    x_val = []
    # x_test = []
    y_train = []
    y_val = []
    # y_test = []

    topic_names = []
    topics_data = {}

    for topic in image_files:
        images_input = []
        images_label = []
        for image_file in topic:
            data_input = open_and_convert(interpolated_dir_name, image_file)
            data_label = open_and_convert(original_dir_name, image_file)

            if data_input.shape != (3, 256, 256):
                print('Wrong input size: {}, {}'.format(image_file, data_input.shape))
                continue
            if data_label.shape != (3, 256, 256):
                print('Wrong label size: {}, {}'.format(image_file, data_label.shape))
                continue

            # split image into small patches
            patches_input = extract_patches(data_input)
            patches_label = extract_patches(data_label)

            images_input.extend(patches_input)
            images_label.extend(patches_label)

        n = len(images_input)
        training_size = round(n * training_r)
        test_size = round(n * test_r)
        validation_size = n - training_size - test_size

        x_train.extend(images_input[:training_size])
        x_val.extend(images_input[training_size:(training_size + validation_size)])
        # x_test.extend(images_input[training_size + validation_size:])

        y_train.extend(images_label[:training_size])
        y_val.extend(images_label[training_size:(training_size + validation_size)])
        # y_test.extend(images_label[training_size + validation_size:])

        topic_name = Path(topic[0]).parent.name
        if topic_name not in topics_data:
            topic_names.append(topic_name)
            topics_data[topic_name] = [[], []]
        topics_data[topic_name][0].extend(images_input[training_size + validation_size:])
        topics_data[topic_name][1].extend(images_label[training_size + validation_size:])

    with h5py.File('../dataset/split.h5', 'w') as f:
        f.create_dataset('x_train', data=np.array(x_train)[:, :1])
        f.create_dataset('x_val', data=np.array(x_val)[:, :1])
        # f.create_dataset('x_test', data=np.array(x_test)[:, :1)
        f.create_dataset('y_train', data=np.array(y_train)[:, :1])
        f.create_dataset('y_val', data=np.array(y_val)[:, :1])
        # f.create_dataset('y_test', data=np.array(y_test)[:, :1)

    with h5py.File('../dataset/test.h5', 'w') as f:
        for topic_name in topic_names:
            topics_data[topic_name][0] = np.array(topics_data[topic_name][0])[:, :1]
            topics_data[topic_name][1] = np.array(topics_data[topic_name][1])[:, :1]
            f.create_dataset(topic_name, data=np.array(topics_data[topic_name]))

    with open('../dataset/topics.json', 'w') as f:
        f.write(json.dumps(topic_names))


def open_and_convert(dir_name, relative_pathfile):
    img = Image.open(Path(root_dir, dir_name, relative_pathfile)).convert('YCbCr')
    t = TF.to_tensor(img)
    data = t.detach().numpy()
    return data


def extract_patches(data):
    nw = data.shape[2] // patch_width
    nh = data.shape[1] // patch_height
    patches = []
    for i in range(nh):
        for j in range(nw):
            start_i = i * patch_height
            start_j = j * patch_width
            patches.append(data[:, start_i:start_i + patch_height, start_j:start_j + patch_width])
    return patches


if __name__ == '__main__':
    train_test_split()
