import os
import sys
from os.path import exists

import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger

import tensorflow_quantum as tfq

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from classical.autoencoderModels import *
from quantum.embeddings import angle_embedding
from quantum.fvqc import create_fvqc
from quantum.gvqc import create_gvqc
from quantum.mera import create_mera
from quantum.svqc import create_svqc
from utils import batch_encode_array, unique2D_subarray, hinge_accuracy, flatten_data, parse_args, organize_data_ovr


def dim_reduc(dataset, train_layer, train_features, test_features, val_features, base_dir, train_count, test_count, val_count): 
    
    print('Starting dimensional reduction with VGG16 and autoencoder!')

    latent_dim = 16
    
    if dataset == 'eurosat':
        if train_layer != 'dense':
            
            x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count, test_count,
                                                  val_count)
            autoencoder = DeepAutoencoder_64(latent_dim)
            
        if train_layer == 'dense':
            x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count, test_count,
                                                  val_count)
            autoencoder = DeepAutoencoder_64(latent_dim)
    
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
            
        
    if dataset == 'resisc45':
        if train_layer == 'fvqc' or train_layer == 'svqc':
            image_size = [256, 256, 3]
            classes = ['storage_tank', 'beach', 'palace', 'airport', 'dense_residential', 'tennis_court', 'thermal_power_station', 'ship', 'chaparral', 'bridge', 'snowberg', 'roundabout', 'commercial_area', 'sea_ice', 'meadow', 'intersection', 'basketball_court', 'golf_course', 'ground_track_field', 'desert', 'railway_station', 'mobile_home_park', 'parking_lot', 'island', 'airplane', 'harbor', 'cloud', 'mountain', 'industrial_area', 'forest', 'rectangular_farmland', 'medium_residential', 'church', 'overpass', 'freeway', 'baseball_diamond', 'river', 'wetland', 'railway', 'runway', 'lake', 'stadium', 'circular_farmland', 'terrace', 'sparse_residential']

            x_train = train_features.reshape(-1, image_size[0], image_size[1], image_size[2])
            x_test = test_features.reshape(-1, image_size[0], image_size[1], image_size[2])
            x_val = val_features.reshape(-1, image_size[0], image_size[1], image_size[2])
            
            train_dir = os.path.join(base_dir, 'train')
            test_dir = os.path.join(base_dir, 'test')
            val_dir = os.path.join(base_dir, 'valid')
            
            train_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(train_dir,
                                                                 target_size=(image_size[0], image_size[1]),
                                                                 batch_size=32,
                                                                 classes = classes,
                                                                 class_mode='input')
            
            test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(test_dir,
                                                                 target_size=(image_size[0], image_size[1]),
                                                                 batch_size=32,
                                                                 classes = classes,
                                                                 class_mode='input')
            
            autoencoder = ConvAutoencoder_256(latent_dim, image_size)
            
            autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
            autoencoder.fit(train_generator,
                            epochs=50,
                            shuffle=True,
                            validation_data=test_generator,
                            workers=multiprocessing.cpu_count()
                            )

            encoded_x_train_ = batch_encode_array(autoencoder, x_train, 441)  # 22050/50
            encoded_x_test_ = batch_encode_array(autoencoder, x_test, 189)  # 4725/25
            encoded_x_val_ = batch_encode_array(autoencoder, x_val, 189)
        
        if train_layer == 'gvqc':
            print('Starting dimensional reduction with convolutional autoencoder!')

            x_train = train_features.reshape(-1, image_size[0], image_size[1], image_size[2])
            x_test = test_features.reshape(-1, image_size[0], image_size[1], image_size[2])
            x_val = val_features.reshape(-1, image_size[0], image_size[1], image_size[2])
            
            autoencoder = ConvAutoencoder_256(latent_dim, image_size)

            autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())

            train_dir = os.path.join(base_dir, 'train')
            test_dir = os.path.join(base_dir, 'test')
            val_dir = os.path.join(base_dir, 'valid')
            
            train_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(train_dir,
                                                                 target_size=(image_size[0], image_size[1]),
                                                                 batch_size=64,
                                                                 classes = classes,
                                                                 class_mode='input')
    
            test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(test_dir,
                                                                 target_size=(image_size[0], image_size[1]),
                                                                 batch_size=64,
                                                                 classes = classes,
                                                                 class_mode='input')
        
            autoencoder.fit(train_generator,
                            epochs=10,
                            shuffle=True,
                            validation_data=test_generator,
                            workers=multiprocessing.cpu_count()
                            )

            encoded_x_train_ = batch_encode_array(autoencoder, x_train, 441)  # 22050/50
            encoded_x_test_ = batch_encode_array(autoencoder, x_test, 189)  # 4725/25
            encoded_x_val_ = batch_encode_array(autoencoder, x_val, 189)

    encoded_x_train = encoded_x_train_.reshape(int(train_count), 4, 4)
    encoded_x_test = encoded_x_test_.reshape(int(test_count), 4, 4)
    encoded_x_val = encoded_x_val_.reshape(int(val_count), 4, 4)

    enc_x_train_u = unique2D_subarray(encoded_x_train)
    enc_x_test_u = unique2D_subarray(encoded_x_test)
    enc_x_val_u = unique2D_subarray(encoded_x_val)
    print("Encoded unique arrays: Train", enc_x_train_u.shape, "and: Test", enc_x_test_u.shape, "and: Val",
          enc_x_val_u.shape)

    return encoded_x_train, encoded_x_test, encoded_x_val
    
    
def quantum_embedding(train_layer, encoded_x_train, encoded_x_test, encoded_x_val):
    if train_layer == 'dense':
        x_train_tfcirc = encoded_x_train
        x_test_tfcirc = encoded_x_test
        x_val_tfcirc = encoded_x_val

    if train_layer == 'mera':
        eparam = 'y'
        print(eparam, 'Angle embedding!')
        train_maximum = np.max(np.abs(encoded_x_train))
        test_maximum = np.max(np.abs(encoded_x_test))
        val_maximum = np.max(np.abs(encoded_x_val))
        x_train_norm = encoded_x_train / train_maximum
        x_test_norm = encoded_x_test / test_maximum
        x_val_norm = encoded_x_val / val_maximum

        x_train_circ = [angle_embedding(x, eparam) for x in x_train_norm]
        x_test_circ = [angle_embedding(x, eparam) for x in x_test_norm]
        x_val_circ = [angle_embedding(x, eparam) for x in x_val_norm]
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
        x_val_tfcirc = tfq.convert_to_tensor(x_val_circ)

    if train_layer == 'fvqc':
        eparam = 'y'
        print(eparam, 'Angle embedding!')
        train_maximum = np.max(np.abs(encoded_x_train))
        test_maximum = np.max(np.abs(encoded_x_test))
        val_maximum = np.max(np.abs(encoded_x_val))
        x_train_norm = encoded_x_train / train_maximum
        x_test_norm = encoded_x_test / test_maximum
        x_val_norm = encoded_x_val / val_maximum

        x_train_circ = [angle_embedding(x, eparam) for x in x_train_norm]
        x_test_circ = [angle_embedding(x, eparam) for x in x_test_norm]
        x_val_circ = [angle_embedding(x, eparam) for x in x_val_norm]
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
        x_val_tfcirc = tfq.convert_to_tensor(x_val_circ)
        
    if train_layer == 'gvqc':
        eparam = 'x'
        print(eparam, 'Angle embedding!')
        train_maximum = np.max(np.abs(encoded_x_train))
        test_maximum = np.max(np.abs(encoded_x_test))
        val_maximum = np.max(np.abs(encoded_x_val))
        x_train_norm = encoded_x_train / train_maximum
        x_test_norm = encoded_x_test / test_maximum
        x_val_norm = encoded_x_val / val_maximum

        x_train_circ = [angle_embedding(x, eparam) for x in x_train_norm]
        x_test_circ = [angle_embedding(x, eparam) for x in x_test_norm]
        x_val_circ = [angle_embedding(x, eparam) for x in x_val_norm]
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
        x_val_tfcirc = tfq.convert_to_tensor(x_val_circ)
        
    return x_train_tfcirc, x_test_tfcirc, x_val_tfcirc
    
    
def train(dataset, log_path, one_class, rest_classes, x_train_tfcirc, x_test_tfcirc, x_val_tfcirc, y_train, y_test, y_val, train_layer, batchsize2):  
    """LOGGING"""
    csv_logger = CSVLogger(log_path + '/model_log_' + str(one_class) + '.csv', append=True, separator=';')

    """PREPARATION"""
    EPOCHS = 3
    
    if train_layer != 'dense':
        y_train = 2 * y_train - 1
        y_test = 2 * y_test - 1
        y_val = 2 * y_val - 1

    """MODEL CREATION AND TRAINING"""
    if train_layer == 'fvqc':
        observable = 'x'
        circuit, readout = create_fvqc(observable)

    if train_layer == 'gvqc':
        observable = 'x'
        circuit, readout = create_gvqc(observable)

    if train_layer == 'mera':
        observable = 'x'
        circuit, readout = create_mera(observable)

    if train_layer == 'svqc':
        observable = 'x'
        circuit, readout = create_svqc(observable)

    if train_layer == 'dense':
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(4, 4, 1)),
            tf.keras.layers.Dense(2, activation='relu'),
            tf.keras.layers.Dense(1),
        ])

    if train_layer != 'dense':
        model = tf.keras.Sequential([
            # The input is the data-circuit, encoded as a tf.string
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            # The PQC layer returns the expected value of the readout gate, range [-1,1].
            tfq.layers.PQC(circuit, readout),
        ])

    if train_layer != 'dense':
        print('Squared hinge loss selected!')
        model_loss = tf.keras.losses.SquaredHinge()

    if train_layer == 'dense':
        print('Binary cross entropy loss selected!')
        model_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.0)

    model_optimizer = tf.keras.optimizers.Adam()
    
    print('Compiling model .....')
    model.compile(
        loss=model_loss,
        optimizer=model_optimizer,
        metrics=[hinge_accuracy])

    qnn_history = model.fit(
        x_train_tfcirc, y_train,
        batch_size=batchsize2,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test_tfcirc, y_test),
        callbacks=[csv_logger])

    time_4 = time.time()
    print('Model training finished at time 4:', time_4)

    """EVALUATION"""
    qnn_results = model.evaluate(x_val_tfcirc, y_val)
    print(qnn_results)
    print('Model evaluated!')

    plt.figure(figsize=(10,5))
    plt.plot(qnn_history.history['hinge_accuracy'], label='qnn accuracy')
    plt.plot(qnn_history.history['val_hinge_accuracy'], label='qnn val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(log_path + '/acc_' + str(one_class) + '.png')
    
    plt.figure(figsize=(10,5))
    plt.plot(qnn_history.history['loss'], label='qnn loss')
    plt.plot(qnn_history.history['val_loss'], label='qnn val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('val_loss')
    plt.legend()
    plt.savefig(log_path + '/loss_' + str(one_class) + '.png')

    model.save_weights(log_path + '/weights_' + str(one_class) + '.h5')
    print('Model weights saved!')
    
    y_true_ = y_val
    y_pred = model.predict(x_val_tfcirc)
    
    print(y_true_)
    print(y_pred)
    
    if train_layer != 'dense':
        # Hinge labels to 0,1
        y_true_ = (y_true_ + 1)/2
        y_pred = (np.array(y_pred) + 1)/2
 
        y_true = []
        for i in range(len(y_true_)):
            y_true.append(int(y_true_[i]))
        y_true = np.asarray(y_true)

        # Round Labels for Metrics
        y_pred_int = []
        for i in range(0, len(y_pred)):    
            y_pred_int.append(int(round(y_pred[i][0])))

        precision_0 = precision_score(y_true, y_pred_int, pos_label=0, average='binary')
        recall_0 = recall_score(y_true, y_pred_int, pos_label=0, average='binary')
        f1_0 = f1_score(y_true, y_pred_int, pos_label=0, average='binary')

        precision_1 = precision_score(y_true, y_pred_int, pos_label=1, average='binary')
        recall_1 = recall_score(y_true, y_pred_int, pos_label=1, average='binary')
        f1_1 = f1_score(y_true, y_pred_int, pos_label=1, average='binary')

        print('Precision for class ', one_class ,' is: ', precision_0)
        print('Recall for class ', one_class ,' is: ', recall_0)
        print('F1 for class ', one_class ,' is: ', f1_0)

        print('Precision for class for 0 labels is: ', precision_1)
        print('Recall for 0 labels is: ', recall_1)
        print('F1 for 0 labels is: ', f1_1)

        #tmp = set(y_true)-set(y_pred_int)
        #print('Values from set(y_true)-set(y_pred_int):', tmp)
    
    print('-----------------------Training of model for ', str(one_class) , ' finished!-----------------------')
    
    return model 


def extract_features(dataset, directory, sample_count, image_size, classes, vgg16, batchsize1):
    if dataset == 'eurosat':
        if vgg16:
            features = np.zeros(shape=(sample_count, 2, 2, 512))
        if not vgg16:
            features = np.zeros(shape=(sample_count, image_size[0], image_size[1], image_size[2]))
    if dataset == 'resisc45':
        if vgg16:
            features = np.zeros(shape=(sample_count, 8, 8, 512))
        if not vgg16:
            features = np.zeros(shape=(sample_count, image_size[0], image_size[1], image_size[2]))

    labels = np.zeros(shape=(sample_count, len(classes)))
    
    if vgg16:
        conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

    generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(directory,
                                                                         target_size=(image_size[0], image_size[1]),
                                                                         batch_size=batchsize1,
                                                                         classes = classes,
                                                                         class_mode='categorical')

    i = 0

    print('Entering for loop...')

    for inputs_batch, labels_batch in generator:

        if vgg16:
            features_batch = conv_base.predict(inputs_batch)
        if not vgg16:
            features_batch = inputs_batch
        features[i * batchsize1: (i + 1) * batchsize1] = features_batch
        
        labels[i * batchsize1: (i + 1) * batchsize1] = labels_batch
       
        i += 1
        if i * batchsize1 >= sample_count:
            break

    return features, labels, generator.class_indices


def train_ovr(args):
    if args.dataset == 'resisc45':
        train_layer = 'fvqc'
        classes = ['storage_tank', 'beach', 'palace', 'airport', 'dense_residential', 'tennis_court', 'thermal_power_station', 'ship', 'chaparral', 'bridge', 'snowberg', 'roundabout', 'commercial_area', 'sea_ice', 'meadow', 'intersection', 'basketball_court', 'golf_course', 'ground_track_field', 'desert', 'railway_station', 'mobile_home_park', 'parking_lot', 'island', 'airplane', 'harbor', 'cloud', 'mountain', 'industrial_area', 'forest', 'rectangular_farmland', 'medium_residential', 'church', 'overpass', 'freeway', 'baseball_diamond', 'river', 'wetland', 'railway', 'runway', 'lake', 'stadium', 'circular_farmland', 'terrace', 'sparse_residential']
        image_count = np.ones(45)*700
        image_size = [256, 256, 3]
        split = 0.3
        vgg16 = False

    if args.dataset == 'eurosat':
        train_layer = 'fvqc'
        classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
        image_count = [3000, 3000, 3000, 2500, 2500, 2000, 2500, 3000, 2500, 3000]
        image_size = [64, 64, 3]
        split = 0.3
        vgg16 = True
        
    """LOGGING"""
    try:
        os.mkdir('./logs')
    except FileExistsError:
        print('Log directory exists!')

    log_path = os.path.join('./logs/RUN_OneVsRest_' + str(args.dataset) + '_' + str(train_layer))

    k = 0
    try:
        os.mkdir(log_path)
    except FileExistsError:
        while exists(log_path):
            log_path = os.path.join('./logs/RUN_OneVsRest_' + str(args.dataset) + '_' + str(train_layer) + '_' + str(k))
            k+=1
        os.mkdir(log_path)
    sys.stdout = open(log_path + '/output_log.txt', 'w')
    
    start = time.time()
    print('Started at time:', start)
    
    """PREPARATION"""
    organize_data_ovr(dataset_name=args.dataset, input_path=args.dataset_path, classes=classes, split=split)
        
    base_dir = '../' + args.dataset + '_data_OvR'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    val_dir = os.path.join(base_dir, 'valid')

    train_count = 0
    test_count = 0
    val_count = 0
    for i in range(len(classes)):
        train_count += image_count[i] - image_count[i] * split
        test_count += (image_count[i] * split)/2
        val_count += (image_count[i] * split)/2

    train_features, train_labels_, train_class_indices = extract_features(args.dataset, train_dir, int(train_count), image_size, classes, vgg16, args.batchsize1)
    test_features, test_labels_, test_class_indices = extract_features(args.dataset, test_dir, int(test_count), image_size, classes, vgg16, args.batchsize1)
    val_features, val_labels_, val_class_indices = extract_features(args.dataset, val_dir, int(val_count), image_size, classes, vgg16, args.batchsize1)

    print('Total Number of TRAIN images is:' + str(len(train_features)))
    print('Total Number of TEST images is:' + str(len(test_features)))
    print('Total Number of VALIDATION images is:' + str(len(val_features)))
    
    train_labels = []
    for i in range(len(train_labels_)):
        train_labels.append(np.argmax(train_labels_[i]))
        
    test_labels = []
    for i in range(len(test_labels_)):
        test_labels.append(np.argmax(test_labels_[i]))

    val_labels = []
    for i in range(len(val_labels_)):
        val_labels.append(np.argmax(val_labels_[i]))
            
    time_1 = time.time()
    print('Preparation finished at time 1:', time_1)
    
    """DATA PREPROCESSING"""
    x_train, x_test, x_val = dim_reduc(args.dataset, train_layer, train_features, test_features, val_features, base_dir, int(train_count), int(test_count), int(val_count))
        
    time_2 = time.time()
    print('Dimensionality reduction finished at time 2:', time_2)
    
    x_train_tfcirc, x_test_tfcirc, x_val_tfcirc = quantum_embedding(train_layer, x_train, x_test, x_val)
    
    time_3 = time.time()
    print('Quantum embedding finished at time 3:', time_3)
    
    """TRAINING"""
    models = []
    for one_class in classes:
        rest_classes = classes[:]
        rest_classes.remove(one_class)
        
        one_class_int = train_class_indices[one_class]

        # one class as positive, all other classes as negative
        y_train = []
        y_test = []
        y_val = []
        for i in range(len(train_labels)):
            if train_labels[i] == one_class_int:
                y_train.append(1)
            if train_labels[i] != one_class_int:
                y_train.append(0)

        for i in range(len(test_labels)):
            if test_labels[i] == one_class_int:
                y_test.append(1)
            if test_labels[i] != one_class_int:
                y_test.append(0)
                
        for i in range(len(val_labels)):
            if val_labels[i] == one_class_int:
                y_val.append(1)
            if val_labels[i] != one_class_int:
                y_val.append(0)
        
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        y_val = np.asarray(y_val)

        model = train(args.dataset, log_path, one_class, rest_classes, x_train_tfcirc, x_test_tfcirc, x_val_tfcirc, y_train, y_test, y_val, train_layer, args.batchsize2)
        models.append(model)
        
        time_temp = time.time()
        print('A model finished training at:', time_temp)
    
    time_4 = time.time()
    print('All models finished training at time 4:', time_4)

    """EVALUATION"""
    preds = []
    for model in models:
        pred = model.predict(x_val_tfcirc)
        preds.append(pred)  # [model1_pred, model2_pred, ...]
        
    preds_sorted = []
    for i in range(len(x_val_tfcirc)):
        models_together = []
        for pred in preds:
            models_together.append(pred[i])
        preds_sorted.append(models_together)
        
    ovr_pred_values = []
    ovr_preds = []
    for pred in preds_sorted:  
        ovr_pred_values.append(max(pred)) 
        ovr_preds.append(np.argmax(pred)) # The index of the model with the highest value for the predicted class
    
    
    set_test = set(val_labels)-set(ovr_preds)
    print('set(val_labels)-set(ovr_preds)', set_test)
    
    cm = confusion_matrix(val_labels, ovr_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.savefig(log_path + '/confusion_matrix_' + str(train_layer) + str(args.dataset) + '.png')
    print('Confusion Matrix')
    print(classes)
    
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    print('Overall:[ FP, FN, TP, TN ] [', FP, FN, TP, TN, ']')

    TPR = TP/(TP+FN)     # Sensitivity, hit rate, recall, or true positive rate
    print('Sensitivity, hit rate, recall, or true positive rate: ', TPR)

    TNR = TN/(TN+FP)     # Specificity or true negative rate
    print('Specificity or true negative rate: ', TNR)

    PPV = TP/(TP+FP)    # Precision or positive predictive value
    print('Precision or positive predictive value: ', PPV)

    NPV = TN/(TN+FN)    # Negative predictive value
    print('Negative predictive value: ', NPV)

    FPR = FP/(FP+TN)    # Fall out or false positive rate
    print('Fall out or false positive rate: ', FPR)

    FNR = FN/(TP+FN)    # False negative rate
    print('False negative rate: ', FNR)

    FDR = FP/(TP+FP)    # False discovery rate
    print('False discovery rate: ', FDR)

    ACC = (TP+TN)/(TP+FP+FN+TN)    # Overall accuracy
    print('Overall accuracy: ', ACC)

if __name__ == '__main__':
    args = parse_args()
    train_ovr(args)
