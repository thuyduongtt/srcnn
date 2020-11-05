import json

import h5py
import torch
import torch.utils.data as utils
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np

from src.evaluation_metrics import PSNR, RMSE, UIQ, SRE
from src.srcnn import SRCNN
from src.train import SRCNNDataset

batch_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_dataset():
    with open('../dataset/topics.json', 'r') as f:
        topic_names = json.load(f)

    topics = []
    with h5py.File('../dataset/test.h5') as f:
        for topic_name in topic_names:
            topics.append(f[topic_name][:])

    return topic_names, topics


def test_topic(topic, print_output=False):
    print('\n\n=====')
    x_test, y_test = topic
    test_data = SRCNNDataset(x_test, y_test)
    test_loader = utils.DataLoader(test_data, batch_size=batch_size)

    model = SRCNN().to(device)
    model.load_state_dict(torch.load('../output/model.pth'))

    bicubic_metrics = np.zeros(4)
    srcnn_metrics = np.zeros(4)

    it = 0
    model.eval()
    with torch.no_grad():
        for bi, data in tqdm(enumerate(test_loader), total=int(len(test_data) / test_loader.batch_size)):
            image_data = data[0].to(device)
            labels = data[1].to(device)

            bicubic_metrics[0] += PSNR(image_data, labels)
            bicubic_metrics[1] += SRE(image_data, labels)
            bicubic_metrics[2] += RMSE(image_data, labels)
            bicubic_metrics[3] += UIQ(image_data, labels)

            outputs = model(image_data)
            srcnn_metrics[0] += PSNR(outputs, labels)
            srcnn_metrics[1] += SRE(outputs, labels)
            srcnn_metrics[2] += RMSE(outputs, labels)
            srcnn_metrics[3] += UIQ(outputs, labels)

            if print_output:
                outputs = outputs.cpu()
                save_image(outputs, f"../output/test_{it}.png")
            it += 1

    b = int(len(test_data) / test_loader.batch_size)
    bicubic_metrics = bicubic_metrics / b
    srcnn_metrics = srcnn_metrics / b
    return bicubic_metrics, srcnn_metrics


def test():
    topic_names, topics = read_dataset()

    for i in range(len(topic_names)):
        bicubic_metrics, srcnn_metrics = test_topic(topics[i])

        print('\n')
        print(topic_names[i])

        for j in bicubic_metrics:
            print(j)
        print('\n')
        for k in srcnn_metrics:
            print(k)


if __name__ == '__main__':
    test()
