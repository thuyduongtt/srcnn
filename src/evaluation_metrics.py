import math

import numpy as np


def SRE(outputs, labels):
    """
    Signal to reconstruction error ratio (the higher the better)
    """
    outputs = outputs.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    dim = len(outputs.shape)
    n = outputs.shape[dim - 2] * outputs.shape[dim - 1]
    mu = np.mean(labels)
    dist = np.linalg.norm(outputs - labels)
    if dist == 0:
        return 0
    return 10 * np.log10(n * (mu ** 2) / dist)


def RMSE(outputs, labels):
    """
    Root mean squared error (the lower the better)
    """
    outputs = outputs.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    diff = outputs - labels
    return math.sqrt(np.mean(diff ** 2))


def PSNR(outputs, labels, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better)
    """
    r = RMSE(outputs, labels)
    if r == 0:
        return 100
    else:
        psnr = 20 * math.log10(max_val / r)
        return psnr


def UIQ(outputs, labels, data_range=1.):
    """
    Compute Universal image quality index (UIQ) (the higher the better)
    """
    outputs = outputs.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    n = outputs.shape[0]
    h = outputs.shape[2]
    w = outputs.shape[3]

    win_size = 7
    final_Q = 0.0

    for i in range(n):
        o = outputs[i][0]
        l = labels[i][0]
        current_Q = 0.0
        m = 0
        for y in range(h - win_size + 1):
            for x in range(w - win_size + 1):
                current_Q += uiq_window(o[y:y + win_size, x:x + win_size], l[y:y + win_size, x:x + win_size], win_size)
                m += 1
        current_Q /= m
        final_Q += current_Q

    final_Q /= n
    return final_Q


def uiq_window(output, label, win_size):
    output_mean = output.mean()
    label_mean = label.mean()

    n = win_size ** 1
    _1n1 = 1 / (n - 1)
    output_sigma = _1n1 * np.sum((output - output_mean) ** 2)
    label_sigma = _1n1 * np.sum((label - label_mean) ** 2)
    output_label_sigma = _1n1 * np.sum((output - output_mean) * (label - label_mean))

    top = 4 * output_label_sigma * output_mean * label_mean
    bot = (output_sigma + label_sigma) * (output_mean ** 2 + label_mean ** 2)

    return top / bot
