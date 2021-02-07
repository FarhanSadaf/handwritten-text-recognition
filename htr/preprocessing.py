import cv2
import numpy as np


def preprocess(img, input_size):
    '''
    "Preprocesses an image of word."
    img: Grayscale (H, W)
    Reads image from 'imgpath';
    img (H, W, C) can also be given
        reshapes it according to its ratio;
        fill extra spots by image background; 
        transposes cv2 image.
    Returns transposed image of size 'input_size'.

    Preprocess metodology based in:
        H. Scheidl, S. Fiel and R. Sablatnig,
        Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm, in
        16th International Conference on Frontiers in Handwriting Recognition, pp. 256-258, 2018.
    '''
    wt, ht, _ = input_size

    h, w = img.shape
    f = max((w / wt), (h / ht))
    new_shape = max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)    # (W, H)

    img = cv2.resize(img, new_shape)    # img.shape : (H, W)

    # As img_shape = (1024, 128, 1) and new img shape = (637, 128)
    # so, we fill remaining spots with 255
    target = np.ones((ht, wt), dtype=np.uint8) * 255
    target[:new_shape[1], :new_shape[0]] = img      # target.shape : (input_size[1], input_size[0])

    img = cv2.transpose(target)     # img.shape : (input_size[0], input_size[1])
    return img


def augmentation(imgs,
                 rotation_range=0,
                 scale_range=0,
                 height_shift_range=0,
                 width_shift_range=0,
                 dilate_range=1,
                 erode_range=1):
    '''
    Apply variations to a list of images (rotate, width and height shift, scale, erode, dilate)
    '''
    imgs = imgs.astype(np.float32)
    _, h, w = imgs.shape

    dilate_kernel = np.ones((int(np.random.uniform(1, dilate_range)),), np.uint8)
    erode_kernel = np.ones((int(np.random.uniform(1, erode_range)),), np.uint8)
    height_shift = np.random.uniform(-height_shift_range, height_shift_range)
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - scale_range, 1)
    width_shift = np.random.uniform(-width_shift_range, width_shift_range)

    trans_map = np.float32([[1, 0, width_shift * w], [0, 1, height_shift * h]])
    rot_map = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)

    trans_map_aff = np.r_[trans_map, [[0, 0, 1]]]
    rot_map_aff = np.r_[rot_map, [[0, 0, 1]]]
    affine_mat = rot_map_aff.dot(trans_map_aff)[:2, :]

    for i in range(len(imgs)):
        imgs[i] = cv2.warpAffine(imgs[i], affine_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
        imgs[i] = cv2.erode(imgs[i], erode_kernel, iterations=1)
        imgs[i] = cv2.dilate(imgs[i], dilate_kernel, iterations=1)

    return imgs


def normalization(imgs):
    '''
    Accepts img (1024, 128) : list
    Returns imgs (n, 1024, 128, 1) normalized (1 / 255.0): np.array
    '''
    imgs = np.array(imgs).astype(np.float32)
    imgs = np.expand_dims(imgs / 255.0, axis=-1)
    return imgs


def threshold(img, block_size=25, offset=10):
    '''
    Local gaussian image thresholding.
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, offset)
