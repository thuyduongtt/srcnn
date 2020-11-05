import time
from abc import ABC

import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from torchvision.utils import save_image
from tqdm import tqdm

from src.evaluation_metrics import PSNR
from src.srcnn import SRCNN

# learning parameters
batch_size = 256  # batch size, reduce if facing OOM error
num_workers = 0
epochs = 50  # number of epochs to train the SRCNN model for
lr = 0.00001  # the learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# evaluation metric
eval_metric = PSNR
eval_metric_unit = 'PSNR (dB)'


class SRCNNDataset(utils.Dataset, ABC):
    def __init__(self, input_images, label_images):
        self.input_images = input_images
        self.label_images = label_images

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, index):
        image = self.input_images[index]
        label = self.label_images[index]
        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.float)
        )


def read_dataset():
    with h5py.File('../dataset/split.h5') as f:
        x_train = f['x_train'][:]
        y_train = f['y_train'][:]
        x_val = f['x_val'][:]
        y_val = f['y_val'][:]

    return x_train, y_train, x_val, y_val


def train(model, optimizer, criterion, train_data, dataloader):
    model.train()
    running_loss = 0.0
    running_eval_metric = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
        image_data = data[0].to(device)
        labels = data[1].to(device)

        # zero grad the optimizer
        optimizer.zero_grad()
        outputs = model(image_data)
        loss = criterion(outputs, labels)
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        # add loss of each item (total items in a batch = batch size)
        running_loss += loss.item()
        # calculate batch metric (once every `batch_size` iterations)
        running_eval_metric += eval_metric(outputs, labels)
    final_loss = running_loss / len(dataloader.dataset)
    final_eval_metric = running_eval_metric / int(len(train_data) / dataloader.batch_size)
    return final_loss, final_eval_metric


def validate(model, criterion, train_data, dataloader, epoch):
    model.eval()
    running_loss = 0.0
    running_eval_metric = 0.0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
            image_data = data[0].to(device)
            labels = data[1].to(device)

            outputs = model(image_data)
            loss = criterion(outputs, labels)
            # add loss of each item (total items in a batch = batch size)
            running_loss += loss.item()
            # calculate batch metric (once every `batch_size` iterations)
            running_eval_metric += eval_metric(outputs, labels)
        outputs = outputs.cpu()
        save_image(outputs, f"../output/val_sr{epoch}.png")
    final_loss = running_loss / len(dataloader.dataset)
    final_eval_metric = running_eval_metric / int(len(train_data) / dataloader.batch_size)
    return final_loss, final_eval_metric


def save_plots(train_loss, val_loss, train_eval_metric, val_eval_metric):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../output/loss.png')
    # plt.show()

    # eval metric plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_eval_metric, color='green', label=f"train {eval_metric_unit}")
    plt.plot(val_eval_metric, color='blue', label=f"validation {eval_metric_unit}")
    plt.xlabel('Epochs')
    plt.ylabel(eval_metric_unit)
    plt.legend()
    plt.savefig('../output/eval_metric.png')
    # plt.show()


def start_training():
    x_train, y_train, x_val, y_val = read_dataset()

    train_data = SRCNNDataset(x_train, y_train)
    val_data = SRCNNDataset(x_val, y_val)

    train_loader = utils.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    val_loader = utils.DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)

    model = SRCNN().to(device)

    # optimizer
    optimizer = optim.Adam(
        [
            {"params": model.conv1.parameters(), "lr": 0.0001},
            {"params": model.conv2.parameters(), "lr": 0.0001},
            {"params": model.conv3.parameters(), "lr": 0.00001},
        ], lr=lr)

    # loss function
    criterion = nn.MSELoss()

    train_loss, val_loss = [], []
    train_eval_metric, val_eval_metric = [], []
    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_eval_metric = train(model, optimizer, criterion, train_data,
                                                          train_loader)
        val_epoch_loss, val_epoch_eval_metric = validate(model, criterion, train_data, val_loader, epoch)
        print(f"\nTrain (evaluation): {train_epoch_eval_metric:.3f}")
        print(f"Val (evaluation): {val_epoch_eval_metric:.3f}")
        train_loss.append(train_epoch_loss)
        train_eval_metric.append(train_epoch_eval_metric)
        val_loss.append(val_epoch_loss)
        val_eval_metric.append(val_epoch_eval_metric)
    end = time.time()
    print(f"Finished training in: {((end - start) / 60):.3f} minutes")

    # save the model to disk
    print('Saving model...')
    torch.save(model.state_dict(), '../output/model.pth')

    save_plots(train_loss, val_loss, train_eval_metric, val_eval_metric)


if __name__ == '__main__':
    start_training()
