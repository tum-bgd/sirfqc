import os
import sys
from os.path import exists

import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger

from skimage.transform import downscale_local_mean
from sklearn.decomposition import PCA, FactorAnalysis

from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow_quantum as tfq

from classical.autoencoderModels import *
from classical.dae import DAE
from classical.rbm import train_rbm
from quantum.embeddings import basis_embedding, angle_embedding
from quantum.fvqc import create_fvqc
from quantum.gvqc import create_gvqc
from quantum.mera import create_mera
from quantum.svqc import create_svqc
from utils import *


def train(args):
    latent_dim = 16 # equals number of data qubits
    
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == 'eurosat':
        image_size = [64, 64, 3]

    if args.dataset == 'resisc45':
        image_size = [256, 256, 3]

    try:
        os.mkdir('./logs')
    except FileExistsError:
        print('Log directory exists!')

    log_path = os.path.join('./logs/RUN_' + str(args.dataset) + '_' + str(args.class1) + 'vs' + str(args.class2) + '_' +
                            str(args.preprocessing) + '_' + 'vgg16' + str(args.vgg16) + '_' + str(args.embedding) + str(args.embeddingparam) + '_' +
                            str(args.train_layer) + '_' + str(args.loss) + '_' + str(args.observable))
    k = 0
    try:
        os.mkdir(log_path)
    except FileExistsError:
        while exists(log_path):
            log_path = os.path.join('./logs/RUN_' + str(args.dataset) + '_' + str(args.class1) + 'vs' + str(args.class2) + '_' +
                                str(args.preprocessing) + '_' + 'vgg16' + str(args.vgg16) + '_' + str(args.embedding) + str(args.embeddingparam) + '_' +
                                str(args.train_layer) + '_' + str(args.loss) + '_' + str(args.observable) + '_' + str(k))
            k+=1
        os.mkdir(log_path)

    sys.stdout = open(log_path + '/output_log.txt', 'w')
    csv_logger = CSVLogger(log_path + '/model_log.csv', append=True, separator=';')

    start = time.time()
    print('OA timer started at:', start)

    organize_data(dataset_name=args.dataset, input_path=args.dataset_path, classes=[args.class1, args.class2], split=int(0.3*args.image_count))

    base_dir = './' + '../' + args.dataset + '_data_' + args.class1 + '_' + args.class2
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    val_dir = os.path.join(base_dir, 'valid')

    train_count = args.image_count + args.image_count - int(0.6*args.image_count)
    test_count = int(0.3*args.image_count)
    val_count = test_count

    train_features, train_labels = extract_features(args.dataset, train_dir, train_count, image_size, args.preprocessing, args.vgg16, args.batchsize1)
    test_features, test_labels = extract_features(args.dataset, test_dir, test_count, image_size, args.preprocessing, args.vgg16, args.batchsize1)
    val_features, val_labels = extract_features(args.dataset, val_dir, val_count, image_size, args.preprocessing, args.vgg16, args.batchsize1)

    print('Total Number of ' + str(args.class1) + ' and ' + str(args.class2) + ' TRAIN images is:' +
          str(len(train_features)))
    print('Total Number of ' + str(args.class1) + ' and ' + str(args.class2) + ' TEST images is:' +
          str(len(test_features)))
    print('Total Number of ' + str(args.class1) + ' and ' + str(args.class2) + ' VALIDATION images is:' +
          str(len(val_features)))

    r, c = train_labels.shape

    print('Labels are:' + str(train_labels.shape))

    if c > 2:
        train_labels = shorten_labels(train_labels)
        test_labels = shorten_labels(test_labels)
        val_labels = shorten_labels(val_labels)

    y_train = single_label(train_labels)
    y_test = single_label(test_labels)
    y_val = single_label(val_labels)

    print('Label ok?:' + str(y_train[0]) + 'and' + str(y_train[1]) + 'and' + str(y_train[2]) + 'and' + str(y_train[3]))

    if args.loss == 'hinge' or args.loss == 'squarehinge':
        # convert labels from 1, 0 to 1, -1
        y_train = 2.0 * y_train - 1.0
        y_test = 2.0 * y_test - 1.0
        y_val = 2.0 * y_val - 1.0

    time_1 = time.time()
    passed = time_1 - start
    print('Elapsed time for preperation:', passed)

    """GRAYSCALE"""
    if args.grayscale and args.preprocessing != 'ds':
        print('Images BRG2GRAY')
        x_train = []
        x_test = []
        x_val = []

        k = 0
        for img in train_features:
            x_train.append(0.2125 * np.float32(img[:,:,0]) + 0.7154 * np.float32(img[:,:,1]) + 0.0721 * np.float32(img[:,:,2]))
            k += 1
        k = 0
        for img in test_features:
            x_test.append(0.2125 * np.float32(img[:,:,0]) + 0.7154 * np.float32(img[:,:,1]) + 0.0721 * np.float32(img[:,:,2]))
            k += 1
        k = 0
        for img in val_features:
            x_val.append(0.2125 * np.float32(img[:,:,0]) + 0.7154 * np.float32(img[:,:,1]) + 0.0721 * np.float32(img[:,:,2]))
            k += 1

        train_features = np.asarray(x_train)
        test_features = np.asarray(x_test)
        val_features = np.asarray(x_val)

    """DOWNSAMPLING"""
    if args.preprocessing == 'ds':
        print('Starting dimensional reduction with downsampling!')

        # convert to single illuminance channel
        _, train_s1, train_s2, _ = train_features.shape
        _, test_s1, test_s2, _ = test_features.shape
        _, val_s1, val_s2, _ = val_features.shape

        x_train = np.zeros((train_count, train_s1, train_s2))
        x_test = np.zeros((test_count, test_s1, test_s2))
        x_val = np.zeros((val_count, val_s1, val_s2))

        k = 0
        for img in train_features:
            x_train[k] = 0.2125 * np.float32(img[:,:,0]) + 0.7154 * np.float32(img[:,:,1]) + 0.0721 * np.float32(img[:,:,2])
            k += 1
        k = 0
        for img in test_features:
            x_test[k] = 0.2125 * np.float32(img[:,:,0]) + 0.7154 * np.float32(img[:,:,1]) + 0.0721 * np.float32(img[:,:,2])
            k += 1
        k = 0
        for img in val_features:
            x_val[k] = 0.2125 * np.float32(img[:,:,0]) + 0.7154 * np.float32(img[:,:,1]) + 0.0721 * np.float32(img[:,:,2])
            k += 1

        # Downsampling
        ds_param = int(image_size[0] / 4)

        encoded_x_train = np.zeros((train_count, 4, 4))
        i = 0
        for img in x_train:
            encoded_x_train[i] = downscale_local_mean(img, (ds_param, ds_param))
            i += 1
        encoded_x_test = np.zeros((test_count, 4, 4))
        i = 0
        for img in x_test:
            encoded_x_test[i] = downscale_local_mean(img, (ds_param, ds_param))
            i += 1
        encoded_x_val = np.zeros((val_count, 4, 4))
        i = 0
        for img in x_val:
            encoded_x_val[i] = downscale_local_mean(img, (ds_param, ds_param))
            i += 1

    """PRINCIPAL COMPONENT ANALYSIS"""
    if args.preprocessing == 'pca':
        print('Starting dimensional reduction with PCA!')

        x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count, test_count,
                                              val_count)

        pca = PCA(n_components=16)

        pca.fit(x_train)

        encoded_x_train = pca.transform(x_train)
        encoded_x_test = pca.transform(x_test)
        encoded_x_val = pca.transform(x_val)

        encoded_x_train = encoded_x_train.reshape(train_count, 4, 4)
        encoded_x_test = encoded_x_test.reshape(test_count, 4, 4)
        encoded_x_val = encoded_x_val.reshape(val_count, 4, 4)

    """AUTOENCODER"""
    if args.preprocessing == 'ae':
        x_train, x_test, x_val = flatten_gray_data(train_features, test_features, val_features, train_count, test_count,
                                                   val_count)

        autoencoder = SimpleAutoencoder_64(latent_dim)

        autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
        autoencoder.fit(x_train, x_train,
                        epochs=50,
                        shuffle=True,
                        validation_data=(x_test, x_test),
                        workers=multiprocessing.cpu_count()
                        )

        encoded_x_train_ = autoencoder.encoder(x_train).numpy()
        encoded_x_test_ = autoencoder.encoder(x_test).numpy()
        encoded_x_val_ = autoencoder.encoder(x_val).numpy()

        encoded_x_train = encoded_x_train_.reshape(train_count, 4, 4)
        encoded_x_test = encoded_x_test_.reshape(test_count, 4, 4)
        encoded_x_val = encoded_x_val_.reshape(val_count, 4, 4)

    if args.preprocessing == 'dae':
        if args.vgg16:
            print('Starting dimensional reduction with VGG16 and autoencoder!')

            if args.dataset == 'eurosat':
                x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count,
                                                      test_count,
                                                      val_count)
                autoencoder = DeepAutoencoder_64(latent_dim)

            if args.dataset == 'resisc45':
                x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count,
                                                      test_count,
                                                      val_count)
                autoencoder = SimpleAutoencoder_256(latent_dim)

            autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
            autoencoder.fit(x_train, x_train,
                            epochs=50,
                            shuffle=True,
                            validation_data=(x_test, x_test),
                            workers=multiprocessing.cpu_count()
                            )

            encoded_x_train_ = autoencoder.encoder(x_train).numpy()
            encoded_x_test_ = autoencoder.encoder(x_test).numpy()
            encoded_x_val_ = autoencoder.encoder(x_val).numpy()

        if not args.vgg16:
            print('Starting dimensional reduction with convolutional autoencoder!')

            x_train = train_features.reshape(-1, image_size[0], image_size[1], image_size[2])
            x_test = test_features.reshape(-1, image_size[0], image_size[1], image_size[2])
            x_val = val_features.reshape(-1, image_size[0], image_size[1], image_size[2])

            if image_size[0] == 256:
                autoencoder = ConvAutoencoder_256(latent_dim, image_size)

            if image_size[0] == 64:
                autoencoder = ConvAutoencoder_64(latent_dim, image_size)

            if image_size[0] != 256 and image_size[0] != 64:
                print('No matching autoencoder for image size' + str(image_size[0]) + 'found!')

            autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())

            autoencoder.fit(x_train, x_train,
                            batch_size=args.batchsize1,
                            epochs=10,
                            shuffle=True,
                            validation_data=(x_test, x_test),
                            workers=multiprocessing.cpu_count()
                            )

            encoded_x_train_ = batch_encode_array(autoencoder, x_train, 10)
            encoded_x_test_ = autoencoder.encoder(x_test).numpy()
            encoded_x_val_ = autoencoder.encoder(x_val).numpy()

        encoded_x_train = encoded_x_train_.reshape(train_count, 4, 4)
        encoded_x_test = encoded_x_test_.reshape(test_count, 4, 4)
        encoded_x_val = encoded_x_val_.reshape(val_count, 4, 4)

    """RBM AUTOENCODER"""
    if args.preprocessing == 'rbmae':
        print('Starting dimensional reduction with deep autoencoder!')

        seed_everything(42)

        if args.vgg16 and args.dataset == 'eurosat':
            num = 2 * 2 * 512
        if args.vgg16 and args.dataset == 'resisc45':
            num = 8 * 8 * 512
        if not args.vgg16:
            if not args.grayscale:
                num = image_size[0] * image_size[1] * image_size[2]
            if args.grayscale:
                num = image_size[0] * image_size[1]

        if args.grayscale:
            x_train, x_test, x_val = flatten_gray_data(train_features, test_features, val_features, train_count,
                                                       test_count,
                                                       val_count)
        if not args.grayscale:
            x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count, test_count,
                                                  val_count)
        x_train_binary, x_test_binary, x_val_binary = binarization(x_train, x_test, x_val)

        train_dl = DataLoader(
            TensorDataset(torch.Tensor(x_train_binary).to(args.device)),
            batch_size=args.batchsize1,
            shuffle=False
        )

        hidden_dimensions = [
            {
                "hidden_dim": 1000,
                "num_epochs": 10,
                "learning_rate": 0.1,
                "use_gaussian": False
            },
            {
                "hidden_dim": 500,
                "num_epochs": 10,
                "learning_rate": 0.1,
                "use_gaussian": False
            },
            {
                "hidden_dim": 250,
                "num_epochs": 10,
                "learning_rate": 0.1,
                "use_gaussian": False
            },
            {
                "hidden_dim": 16,
                "num_epochs": 30,
                "learning_rate": 0.001,  # Use much lower LR for gaussian to avoid exploding gradient
                "use_gaussian": True  # Use a Gaussian distribution for the last hidden layer to let it take advantage of continuous values
            }
        ]

        new_train_dl = train_dl
        visible_dim = num
        hidden_dim = None
        models = []
        for configs in hidden_dimensions:
            hidden_dim = configs["hidden_dim"]
            num_epochs = configs["num_epochs"]
            lr = configs["learning_rate"]
            use_gaussian = configs["use_gaussian"]

            print(str(visible_dim) + ' to ' + str(hidden_dim))
            model, v, v_pred = train_rbm(new_train_dl, visible_dim, hidden_dim, k=1, num_epochs=num_epochs, lr=lr,
                                         use_gaussian=use_gaussian)
            models.append(model)

            new_data = []
            for data_list in new_train_dl:
                p = model.sample_h(data_list[0])[0]
                new_data.append(p.detach().cpu().numpy())
            new_input = np.concatenate(new_data)
            new_train_dl = DataLoader(
                TensorDataset(torch.Tensor(new_input).to(args.device)),
                batch_size=args.batchsize1,
                shuffle=False
            )

            visible_dim = hidden_dim

        # FINE TUNE AUTOENCODER
        lr = 1e-3
        dae = DAE(models).to(args.device)
        dae_loss = nn.MSELoss()
        optimizer = optim.Adam(dae.parameters(), lr)
        num_epochs = 50

        encoded = []
        # train
        for epoch in range(num_epochs):
            losses = []
            for i, data_list in enumerate(train_dl):
                data = data_list[0]
                v_pred, v_encode = dae(data)
                encoded.append(v_encode)
                batch_loss = dae_loss(data, v_pred)
                losses.append(batch_loss.item())
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            running_loss = np.mean(losses)
            print('Epoch', epoch, ':', running_loss)

        # ENCODE DATA
        encoded_x_train = dae_encoding(x_train_binary, dae, args.device)
        encoded_x_test = dae_encoding(x_test_binary, dae, args.device)
        encoded_x_val = dae_encoding(x_val_binary, dae, args.device)

    if args.preprocessing == 'fa':
        print('Starting dimensional reduction with FACTOR ANALYSIS!')

        x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count, test_count,
                                              val_count)

        fa = FactorAnalysis(n_components=16, svd_method='lapack')

        fa.fit(x_train)

        encoded_x_train = fa.transform(x_train)
        encoded_x_test = fa.transform(x_test)
        encoded_x_val = fa.transform(x_val)

        encoded_x_train = encoded_x_train.reshape(train_count, 4, 4)
        encoded_x_test = encoded_x_test.reshape(test_count, 4, 4)
        encoded_x_val = encoded_x_val.reshape(val_count, 4, 4)

    if args.preprocessing == None:
        print('Please chose a dimensional reduction method! ds, pca, ae, dae')
        return

    time_2 = time.time()
    passed = time_2 - time_1
    print('Elapsed time for data compression:', passed)

    enc_x_train_u = unique2D_subarray(encoded_x_train)
    enc_x_test_u = unique2D_subarray(encoded_x_test)
    enc_x_val_u = unique2D_subarray(encoded_x_val)
    print("Encoded unique arrays: Train", enc_x_train_u.shape, "and: Test", enc_x_test_u.shape, "and: Val",
          enc_x_val_u.shape)

    """QUANTUM EMBEDDING"""
    if args.embedding == 'basis' or args.embedding == 'bin':
        x_train_bin, x_test_bin, x_val_bin = binarization(encoded_x_train, encoded_x_test, encoded_x_val)

        """CHECK HOW MANY UNIQUE ARRAYS ARE LEFT AFTER ENCODING"""
        x_train_u = unique2D_subarray(x_train_bin)
        x_test_u = unique2D_subarray(x_test_bin)
        x_val_u = unique2D_subarray(x_val_bin)
        print("Unique arrays after thresholding: Train", x_train_u.shape, "and: Test", x_test_u.shape, "and: Val",
              x_val_u.shape)

    if args.embedding == 'basis':
        print('Basis embedding!')
        x_train_circ = [basis_embedding(x) for x in x_train_bin]
        x_test_circ = [basis_embedding(x) for x in x_test_bin]
        x_val_circ = [basis_embedding(x) for x in x_val_bin]
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
        x_val_tfcirc = tfq.convert_to_tensor(x_val_circ)

    if args.embedding == 'angle':
        print(args.embeddingparam, 'Angle embedding!')
        train_maximum = np.max(np.abs(encoded_x_train))
        test_maximum = np.max(np.abs(encoded_x_test))
        val_maximum = np.max(np.abs(encoded_x_val))
        x_train_norm = encoded_x_train / train_maximum
        x_test_norm = encoded_x_test / test_maximum
        x_val_norm = encoded_x_val / val_maximum

        x_train_circ = [angle_embedding(x, args.embeddingparam) for x in x_train_norm]
        x_test_circ = [angle_embedding(x, args.embeddingparam) for x in x_test_norm]
        x_val_circ = [angle_embedding(x, args.embeddingparam) for x in x_val_norm]
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
        x_val_tfcirc = tfq.convert_to_tensor(x_val_circ)

    if args.embedding == 'bin':
        print('No embedding!')
        x_train_tfcirc = x_train_bin
        x_test_tfcirc = x_test_bin
        x_val_tfcirc = x_val_bin

    if args.embedding == 'no':
        print('No embedding!')
        x_train_tfcirc = encoded_x_train
        x_test_tfcirc = encoded_x_test
        x_val_tfcirc = encoded_x_val

    if args.embedding == None:
        print('Pleaes choose quantum embedding method! basis, angle, no')
        return

    time_3 = time.time()
    passed = time_3 - time_2
    print('Elapsed time for quantum embedding:', passed)

    """MODEL BUILDING"""
    if args.train_layer == 'fvqc':
        circuit, readout = create_fvqc(args.observable)

    if args.train_layer == 'gvqc':
        circuit, readout = create_gvqc(args.observable)

    if args.train_layer == 'mera':
        circuit, readout = create_mera(args.observable)

    if args.train_layer == 'svqc':
        circuit, readout = create_svqc(args.observable)

    if args.train_layer != 'dense':
        print(circuit)

    if args.train_layer == 'dense':
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(4, 4, 1)),
            tf.keras.layers.Dense(2, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    if args.train_layer == None:
        print('Chose a trainig layer! farhi, grant, dense')
        return

    if args.train_layer != 'dense':
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tfq.layers.PQC(circuit, readout),
        ])

    if args.loss == 'hinge':
        print('Hinge loss selected!')
        model_loss = tf.keras.losses.Hinge()

    if args.loss == 'squarehinge':
        print('Square hinge loss selected!')
        model_loss = tf.keras.losses.SquaredHinge()

    if args.loss == 'crossentropy':
        model_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.0)

    if args.loss == None:
        print('Chose a loss function! hinge, squarehinge')
        return

    if args.optimizer == 'adam':
        model_optimizer = tf.keras.optimizers.Adam()

    if args.optimizer == 'bobyqa':
        model_optimizer = 0

    if args.optimizer == None:
        print('Chose an optimizer!')
        return

    print('Compiling model .....')
    if args.train_layer == 'dense':
        model.compile(
            loss=model_loss,
            optimizer=model_optimizer,
            metrics=['accuracy'])
    if args.train_layer != 'dense': 
        model.compile(
            loss=model_loss,
            optimizer=model_optimizer,
            metrics=[hinge_accuracy])

    qnn_history = model.fit(
        x_train_tfcirc, y_train,
        batch_size=args.batchsize2,
        epochs=args.epochs,
        verbose=1,
        validation_data=(x_test_tfcirc, y_test),
        callbacks=[csv_logger])

    time_4 = time.time()
    passed = time_4 - time_3
    print('Elapsed time for training:', passed)
    passed = time_4 - start
    print('OA elapsed time:', passed)

    print('Model training completed!')

    qnn_results = model.evaluate(x_val_tfcirc, y_val)
    print(qnn_results)
    print('Model evaluated!')

    # save figures for accuracy and loss
    if args.train_layer != 'dense':
        plt.figure(figsize=(10, 5))
        plt.plot(qnn_history.history['hinge_accuracy'], label='Accuracy')
        plt.plot(qnn_history.history['val_hinge_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(log_path + '/acc.png')

    if args.train_layer == 'dense':
        plt.figure(figsize=(10, 5))
        plt.plot(qnn_history.history['accuracy'], label='nn accuracy')
        plt.plot(qnn_history.history['val_accuracy'], label='nn val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(log_path + '/acc.png')

    plt.figure(figsize=(10, 5))
    plt.plot(qnn_history.history['loss'], label='Loss')
    plt.plot(qnn_history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss')
    plt.legend()
    plt.savefig(log_path + '/loss.png')

    model.save_weights(log_path + '/weights.h5')
    print('Model weights saved!')

    y_true = y_val
    y_pred = model.predict(x_val_tfcirc)

    if args.loss == 'hinge' or args.loss == 'squarehinge':
        # Hinge labels to 0,1
        y_true = (y_true + 1) / 2
        y_pred = (np.array(y_pred) + 1) / 2

        # Round Labels for Metrics
        y_pred_int = []
        for i in range(0, len(y_pred)):
            y_pred_int.append(round(y_pred[i][0]))

    if args.loss == 'crossentropy':
        y_true = tf.squeeze(y_true) > 0.5
        y_pred_int = tf.squeeze(y_pred) > 0.5

    precision_0 = precision_score(y_true, y_pred_int, pos_label=0, average='binary')
    recall_0 = recall_score(y_true, y_pred_int, pos_label=0, average='binary')
    f1_0 = f1_score(y_true, y_pred_int, pos_label=0, average='binary')

    precision_1 = precision_score(y_true, y_pred_int, pos_label=1, average='binary')
    recall_1 = recall_score(y_true, y_pred_int, pos_label=1, average='binary')
    f1_1 = f1_score(y_true, y_pred_int, pos_label=1, average='binary')

    print('Precision for class ', args.class1, ' is: ', precision_0)
    print('Recall for class ', args.class1, ' is: ', recall_0)
    print('F1 for class ', args.class1, ' is: ', f1_0)

    print('Precision for class ', args.class2, ' is: ', precision_1)
    print('Recall for class ', args.class2, ' is: ', recall_1)
    print('F1 for class ', args.class2, ' is: ', f1_1)


if __name__ == "__main__":
    args = parse_args()
    train(args)
    
