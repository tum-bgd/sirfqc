import os
import random
import numpy as np
import pandas as pd
import torch
import shutil
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
import argparse


def organize_data(dataset_name, input_path, classes, split):
    image_path = Path(input_path)

    # Get sub directories
    types = os.listdir(image_path)
    print("Categories: ", types)

    # class_1 = types.index(classes[0])
    # class_2 = types.index(classes[1])

    types = [classes[0], classes[1]]
    print("Selected categories: ", types)

    # A list that is going to contain tuples: (type, corresponding image path)
    images = []

    for type in types:
        # Get all the file names
        all_images = os.listdir(image_path / type)
        # Add them to the list
        for image in all_images:
            images.append((type, str(image_path / type) + '/' + image))

    # Build a dataframe
    images = pd.DataFrame(data=images, columns=['category', 'image'], index=None)

    # How many samples for each category are present
    print("Total number of images in the dataset: ", len(images))
    image_count = images['category'].value_counts()
    print("Images in each category: ")
    print(image_count)

    try:
        os.mkdir("../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]))
    except FileExistsError:
        print(str(dataset_name), ' data directory already exists')

    try:
        os.mkdir("../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/train")
        os.mkdir("../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/test")
        os.mkdir("../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/valid")
    except FileExistsError:
        print('train,test,val directories already exist')

    try:
        # Inside the train and validation sub=directories, sub-directories for each catgeory
        os.mkdir(
            "../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/train" + "/" + str(types[0]))
        os.mkdir(
            "../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/train" + "/" + str(types[1]))
        os.mkdir("../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/test" + "/" + str(types[0]))
        os.mkdir("../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/test" + "/" + str(types[1]))
        os.mkdir(
            "../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/valid" + "/" + str(types[0]))
        os.mkdir(
            "../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/valid" + "/" + str(types[1]))
    except FileExistsError:
        print(classes[0], ', ', classes[1], ' directories already exist')
        return

    for category in image_count.index:
        samples = images['image'][images['category'] == category].values
        for i in range(int(split / 2)):
            name = samples[i].split('/')[-1]
            shutil.copyfile(samples[i], './' + "../" + str(dataset_name) + '_data_' + str(classes[0]) + '_' + str(
                classes[1]) + '/test/' + str(category) + '/' + name)
        for i in range(int(split / 2), split):
            name = samples[i].split('/')[-1]
            shutil.copyfile(samples[i], './' + "../" + str(dataset_name) + '_data_' + str(classes[0]) + "_" + str(
                classes[1]) + '/valid/' + str(category) + '/' + name)
        for i in range(split, len(samples)):
            name = samples[i].split('/')[-1]
            shutil.copyfile(samples[i], './' + "../" + str(dataset_name) + '_data_' + str(classes[0]) + '_' + str(
                classes[1]) + '/train/' + str(category) + '/' + name)

    print('Train/Test/Val split and directory creation completed!')


def organize_data_ovr(dataset_name, input_path, classes, split):
    image_path = Path(input_path)

    # Get sub directories
    types = os.listdir(image_path)
    print("Categories: ", types)

    # class_1 = types.index(classes[0])
    # class_2 = types.index(classes[1])

    types = []
    for i in range(len(classes)):
        types.append(classes[i])

    print("Selected categories: ", types)

    # A list that is going to contain tuples: (type, corresponding image path)
    images = []

    for type in types:
        # Get all the file names
        all_images = os.listdir(image_path / type)
        # Add them to the list
        for image in all_images:
            images.append((type, str(image_path / type) + '/' + image))

    # Build a dataframe
    images = pd.DataFrame(data=images, columns=['category', 'image'], index=None)

    # How many samples for each category are present
    print("Total number of images in the dataset: ", len(images))
    image_count = images['category'].value_counts()
    print("Images in each category: ")
    print(image_count)

    splits = image_count * split

    try:
        os.mkdir("../" + str(dataset_name) + "_data_OvR")
    except FileExistsError:
        print(str(dataset_name), ' data directory already exists')

    try:
        os.mkdir("../" + str(dataset_name) + "_data_OvR/train")
        os.mkdir("../" + str(dataset_name) + "_data_OvR/test")
        os.mkdir("../" + str(dataset_name) + "_data_OvR/valid")
    except FileExistsError:
        print('train,test,val directories already exist')
    try:
        for i in range(len(types)):
            os.mkdir("../" + str(dataset_name) + "_data_OvR/train" + "/" + str(types[i]))
        for i in range(len(types)):
            os.mkdir("../" + str(dataset_name) + "_data_OvR/test" + "/" + str(types[i]))
        for i in range(len(types)):
            os.mkdir("../" + str(dataset_name) + "_data_OvR/valid" + "/" + str(types[i]))
    except FileExistsError:
        print(types[i], ' directory already exist')
        return

    k = 0
    for category in image_count.index:
        samples = images['image'][images['category'] == category].values
        for i in range(int(splits[k] / 2)):
            name = samples[i].split('/')[-1]
            shutil.copyfile(samples[i], './' + "../" + str(dataset_name) + '_data_OvR/test/' + str(category) + '/' + name)
        for i in range(int(splits[k] / 2), int(splits[k])):
            name = samples[i].split('/')[-1]
            shutil.copyfile(samples[i], './' + "../" + str(dataset_name) + '_data_OvR/valid/' + str(category) + '/' + name)
        for i in range(int(splits[k]), len(samples)):
            name = samples[i].split('/')[-1]
            shutil.copyfile(samples[i], './' + "../" + str(dataset_name) + '_data_OvR/train/' + str(category) + '/' + name)
        k += 1

    print('Train/Test/Val split and directory creation for OvR completed!')


def extract_features(dataset, directory, sample_count, image_size, preprocessing, vgg16, batchsize1):
    if dataset == 'eurosat':
        if vgg16 and preprocessing != 'ds':
            features = np.zeros(shape=(sample_count, 2, 2, 512))
        if not vgg16:
            features = np.zeros(shape=(sample_count, image_size[0], image_size[1], image_size[2]))
    if dataset == 'resisc45':
        if vgg16 and preprocessing != 'ds':
            features = np.zeros(shape=(sample_count, 8, 8, 512))
        if not vgg16:
            features = np.zeros(shape=(sample_count, image_size[0], image_size[1], image_size[2]))

    labels_2 = np.zeros(shape=(sample_count, 2))
    labels_3 = np.zeros(shape=(sample_count, 3))

    if vgg16 and preprocessing != 'ds':
        conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

    generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(directory,
                                                                         target_size=(image_size[0], image_size[1]),
                                                                         batch_size=batchsize1,
                                                                         class_mode='categorical')

    i = 0

    print('Entering for loop...')

    for inputs_batch, labels_batch in generator:

        if vgg16 and preprocessing != 'ds':
            features_batch = conv_base.predict(inputs_batch)
        if not vgg16:
            features_batch = inputs_batch
        if preprocessing == 'ds':
            features_batch = inputs_batch

        features[i * batchsize1: (i + 1) * batchsize1] = features_batch
        try:
            labels_2[i * batchsize1: (i + 1) * batchsize1] = labels_batch
            labels = labels_2
        except:
            labels_3[i * batchsize1: (i + 1) * batchsize1] = labels_batch
            labels = labels_3

        i += 1
        if i * batchsize1 >= sample_count:
            break

    return features, labels


def shorten_labels(labels):
    labels_ = []

    for i in range(0, len(labels)):
        labels_.append(labels[i][1:])

    return labels_


def single_label(labels_):
    labels = []

    for i in range(0, len(labels_)):
        labels.append(labels_[i][0])
    labels = np.array(labels)

    return labels


def batch_encode_array(autoencoder, array, frac):
    cut = int(len(array) / frac)
    encoded = []
    j = 0

    for i in range(1, frac + 1):
        tmp = autoencoder.encoder(array[j * cut:i * cut]).numpy()
        encoded.append(tmp)
        j = i

    encoded_array = np.asarray(encoded)

    return encoded_array


def unique2D_subarray(a):
    dtype1 = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    b = np.ascontiguousarray(a.reshape(a.shape[0], -1)).view(dtype1)

    return a[np.unique(b, return_index=1)[1]]


def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)


def flatten_data(train_features, test_features, val_features, train_count, test_count, val_count):
    _, train_s1, train_s2, train_s3 = train_features.shape
    _, test_s1, test_s2, test_s3 = test_features.shape
    _, val_s1, val_s2, val_s3 = val_features.shape

    x_train = np.reshape(train_features, (train_count, train_s1 * train_s2 * train_s3))
    x_test = np.reshape(test_features, (test_count, test_s1 * test_s2 * test_s3))
    x_val = np.reshape(val_features, (val_count, val_s1 * val_s2 * val_s3))

    return x_train, x_test, x_val


def flatten_gray_data(train_features, test_features, val_features, train_count, test_count, val_count):
    train_s1, train_s2, train_s3 = train_features.shape
    test_s1, test_s2, test_s3 = test_features.shape
    val_s1, val_s2, val_s3 = val_features.shape

    x_train = np.reshape(train_features, (train_s1, train_s2 * train_s3))
    x_test = np.reshape(test_features, (test_s1, test_s2 * test_s3))
    x_val = np.reshape(val_features, (val_s1, val_s2 * val_s3))

    return x_train, x_test, x_val


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def binarization(encoded_x_train, encoded_x_test, encoded_x_val):
    print('Binarization if inputs...')
    unique_tmp = np.unique(encoded_x_train)
    th = np.median(unique_tmp)

    print("Threshold for Binarization is:", th)
    x_train_bin = np.array(encoded_x_train > th, dtype=np.float32)
    x_test_bin = np.array(encoded_x_test > th, dtype=np.float32)
    x_val_bin = np.array(encoded_x_val > th, dtype=np.float32)

    return x_train_bin, x_test_bin, x_val_bin


def dae_encoding(x_binary, dae, device):
    encoded_x = []
    len_input = len(x_binary)
    input_test = torch.Tensor(x_binary[0:len_input]).to(device)

    transformed, tmp = dae.encode(input_test)

    for i in range(0, len(transformed)):
        encoded_x.append(transformed[i].detach().cpu().numpy())
    encoded_x = np.array(encoded_x).reshape(len_input, 4, 4)

    return encoded_x


def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate a hybrid classical-quantum system')
    parser.add_argument('-da', '--dataset', type=str, default='eurosat', help='select dataset. currently available: eurosat, resisc45')
    parser.add_argument('-dp', '--dataset_path', type=str, default='./2750', help='select dataset path')
    parser.add_argument('-c1', '--class1', type=str, default='AnnualCrop', help='select a class for binary classification')
    parser.add_argument('-c2', '--class2', type=str, default='SeaLake', help='select a class for binary classification')
    parser.add_argument('-ic', '--image_count', type=int, default=3000, help='define number of images')
    parser.add_argument('-b1', '--batchsize1', type=int, default=32, help='batch size for preprocessing')
    parser.add_argument('-b2', '--batchsize2', type=int, default=32, help='batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('-t', '--train_layer', type=str, default='fvqc', help='select a training layer. currently available: farhi, grant, dense')
    parser.add_argument('-v', '--vgg16', type=bool, default=True, help='use vgg16 for prior feature extraction True or False')
    parser.add_argument('-cp', '--cparam', type=int, default=0, help='cparam. currently has no influence')
    parser.add_argument('-em', '--embedding', type=str, default='angle', help='select quantum encoding for the classical input data. currently available: basis, angle, ( and bin for no quantum embedding but binarization')
    parser.add_argument('-emp', '--embeddingparam', type=str, default='x', help='select axis for angle embedding')
    parser.add_argument('-l', '--loss', type=str, default='squarehinge', help='select loss function. currently available: hinge, squarehinge, crossentropy')
    parser.add_argument('-ob', '--observable', type=str, default='x', help='select pauli measurement/ quantum observable')
    parser.add_argument('-op', '--optimizer', type=str, default='adam', help='select optimizer. currently available: adam')
    parser.add_argument('-g', '--grayscale', type=bool, default=False, help='TBD: transform input to grayscale True or False')
    parser.add_argument('-p', '--preprocessing', type=str, default='dae', help='select preprocessing technique. currently available: ds, pca, fa, ae, dae (=convae if vgg16=False), rbmae')
    parser.add_argument('-de', '--device', type=str, default=None, help='torch.Device. either "cpu" or "cuda". default will check by torch.cuda.is_available() ')
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args
