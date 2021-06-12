import h5py
import unicodedata
from itertools import groupby
import numpy as np
from .preprocessing import *


class Tokenizer:
    '''
    Tokenizes text.
    '''

    def __init__(self, charset, max_text_len):
        self.PAD_TK, self.UNK_TK = '¶', '¤'     # PAD token on index 0, so 0 on ouput is first removed as it is PAD
        self.chars = self.PAD_TK + self.UNK_TK + charset

        self.PAD = self.chars.find(self.PAD_TK)
        self.UNK = self.chars.find(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.max_len = max_text_len

    def encode(self, text, encoding='ascii'):
        '''
        Encode 'text' to vector.
        '''
        if isinstance(text, bytes):
            # If text in bytes, convert it to str
            text = text.decode()

        text = unicodedata.normalize('NFKD', text).encode(encoding, 'ignore').decode(encoding)
        text = ' '.join(text.split())

        # 'glittters' -> 'glitÂ¤tÂ¤ters'
        groups = [''.join(group) for _, group in groupby(text)]
        text = ''.join([self.UNK_TK.join(c) if len(c) > 1 else c for c in groups])

        encoded = []
        for c in text:
            index = self.chars.find(c)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.array(encoded[:self.max_len])

    def _remove_tokens(self, text):
        return text.replace(self.PAD_TK, '').replace(self.UNK_TK, '')

    def decode(self, text_enc):
        '''
        Decode vector to text.
        '''
        decoded = ''.join([self.chars[int(x)] for x in text_enc])
        decoded = self._remove_tokens(decoded)

        return decoded


class DataGenerator:
    '''
    Generator for ['train', 'val', 'test'] set.
    '''

    def __init__(self, dataset_path, input_size, batch_size, max_text_len, charset, shuffle=True, encoding='ascii'):
        self.tokenizer = Tokenizer(charset, max_text_len)
        self.input_size = input_size
        self.batch_size = batch_size
        self.__encoding = encoding
        self._size = {}
        self._steps = {}
        self.__index = {}
        self.__shuffle = shuffle

        if self.__shuffle:
            self.dataset = dict()

            with h5py.File(dataset_path, 'r') as hf:
                for key in ['train', 'val', 'test']:
                    self.dataset[key] = dict()
                    self.dataset[key]['imgs'] = np.array(hf[key]['imgs'])
                    self.dataset[key]['gt_texts'] = np.array(hf[key]['gt_texts'])

            self.__arange = np.arange(len(self.dataset['train']['gt_texts']))
            np.random.seed(42)
        else:
            self.dataset = h5py.File(dataset_path, 'r')

        for key in ['train', 'val', 'test']:
            self._size[key] = len(self.dataset[key]['gt_texts'])
            self._steps[key] = np.ceil(self._size[key] / self.batch_size).astype(int)

    def next_train_batch(self):
        '''
        Yields next train batch
        '''
        self.__index['train'] = 0

        while True:
            if self.__index['train'] >= self._size['train']:
                self.__index['train'] = 0

                if self.__shuffle:
                    np.random.shuffle(self.__arange)
                    self.dataset['train']['imgs'] = self.dataset['train']['imgs'][self.__arange]
                    self.dataset['train']['gt_texts'] = self.dataset['train']['gt_texts'][self.__arange]

            start = self.__index['train']
            end = start + self.batch_size
            self.__index['train'] = end

            X_train = self.dataset['train']['imgs'][start:end]   # shape : (n, input_size[0], input_size[1])
            X_train = augmentation(X_train,
                                   rotation_range=1.5,
                                   scale_range=0.05,
                                   height_shift_range=0.025,
                                   width_shift_range=0.05,
                                   erode_range=5,
                                   dilate_range=3)
            X_train = normalization(X_train)       # shape : (n, input_size)

            Y_train = [self.tokenizer.encode(y, self.__encoding) for y in self.dataset['train']['gt_texts'][start:end]]
            Y_train = [np.pad(y, (0, self.tokenizer.max_len - len(y))) for y in Y_train]       # Add 0 padding on right side
            Y_train = np.array(Y_train, dtype=np.int16)

            yield (X_train, Y_train)

    def next_val_batch(self):
        '''
        Yields next validation batch.
        '''
        self.__index['val'] = 0

        while True:
            if self.__index['val'] >= self._size['val']:
                self.__index['val'] = 0

            start = self.__index['val']
            end = start + self.batch_size
            self.__index['val'] = end

            X_val = self.dataset['val']['imgs'][start:end]   # shape : (n, input_size[0], input_size[1])
            X_val = normalization(X_val)       # shape : (n, input_size)

            Y_val = [self.tokenizer.encode(y, self.__encoding) for y in self.dataset['val']['gt_texts'][start:end]]
            Y_val = [np.pad(y, (0, self.tokenizer.max_len - len(y))) for y in Y_val]       # Add 0 padding on right side
            Y_val = np.array(Y_val, dtype=np.int16)

            yield (X_val, Y_val)

    def next_test_batch(self):
        '''
        Yields next test batch.
        '''
        self.__index['test'] = 0

        while True:
            if self.__index['test'] >= self._size['test']:
                self.__index['test'] = 0
                break

            start = self.__index['test']
            end = start + self.batch_size
            self.__index['test'] = end

            X_test = self.dataset['test']['imgs'][start:end]   # shape : (n, input_size[0], input_size[1])
            X_test = normalization(X_test)       # shape : (n, input_size)

            yield X_test
