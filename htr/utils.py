import random
import matplotlib.pyplot as plt


def implt(img, figsize=(6, 3), cmap=None, title=None, axis='on'):
    '''
    Plot an image in notebook.
    '''
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap)
    plt.axis(axis)
    plt.title(title)


def split_data(samples, val_split_size=0.1, test_split_size=0.1, shuffle=True):
    '''
    Sample train, validatation, test set.

    samples: list of samples.
    val_split_size: % of len(samples) in validation set.
    test_split_size: % of len(samples) in test set.
    shuffle: If true, samples randomly.

    Returns: dataset{ 'train': list, 'val': list, 'test': list }
    '''
    samples = samples.copy()
    test_split_len = int(len(samples) * test_split_size)
    val_split_len = int(len(samples) * val_split_size)
    train_split_len = len(samples) - (test_split_len + val_split_len)

    dataset = {}
    if shuffle:
        random.shuffle(samples)
    dataset['train'] = samples[: train_split_len]
    dataset['val'] = samples[train_split_len: (train_split_len + val_split_len)]
    dataset['test'] = samples[(train_split_len + val_split_len):]

    return dataset