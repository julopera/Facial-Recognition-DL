# Dependencies #

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import itertools
import os
import os.path
import shutil
import random
import glob
import warnings
import socket
import json

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout, Reshape, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.python.client import device_lib
from keras.applications import inception_v3, mobilenet, mobilenet_v2, densenet, nasnet, efficientnet, efficientnet_v2, resnet
from sklearn.metrics import confusion_matrix

# Organize data for training and testing #

print(os.getcwd())
os.chdir(r'C:\Users\jlope\Downloads\Computer Files\Python Projects\FC_IMAGES')
print(os.getcwd())
if os.path.isdir('train/juan') is False:
    os.makedirs('train/juan')
    os.makedirs('train/sam')
    os.makedirs('train/faris')
    os.makedirs('train/arbaaz')
    os.makedirs('train/arkam')
    os.makedirs('train/nawazish')
    os.makedirs('train/aaron')
    os.makedirs('train/anthony')
    
    os.makedirs('valid/juan')
    os.makedirs('valid/sam')
    os.makedirs('valid/faris')
    os.makedirs('valid/arbaaz')
    os.makedirs('valid/arkam')
    os.makedirs('valid/nawazish')
    os.makedirs('valid/aaron')
    os.makedirs('valid/anthony')
    
    os.makedirs('test/juan')
    os.makedirs('test/sam')
    os.makedirs('test/faris')
    os.makedirs('test/arbaaz')
    os.makedirs('test/arkam')
    os.makedirs('test/nawazish')
    os.makedirs('test/aaron')
    os.makedirs('test/anthony')
    
    for i in random.sample(glob.glob('*/juan.*'), 100):
        shutil.move(i, 'train/juan')      
    for i in random.sample(glob.glob('*/sam.*'), 100):
        shutil.move(i, 'train/sam')
    for i in random.sample(glob.glob('*/faris.*'), 100):
        shutil.move(i, 'train/faris')      
    for i in random.sample(glob.glob('*/arbaaz.*'), 100):
        shutil.move(i, 'train/arbaaz')
    for i in random.sample(glob.glob('*/arkam.*'), 100):
        shutil.move(i, 'train/arkam')      
    for i in random.sample(glob.glob('*/nawazish.*'), 100):
        shutil.move(i, 'train/nawazish')
    for i in random.sample(glob.glob('*/aaron.*'), 100):
        shutil.move(i, 'train/aaron')      
    for i in random.sample(glob.glob('*/anthony.*'), 100):
        shutil.move(i, 'train/anthony')
    for i in random.sample(glob.glob('*/juan.*'), 35):
        shutil.move(i, 'valid/juan')        
    for i in random.sample(glob.glob('*/sam.*'), 35):
        shutil.move(i, 'valid/sam')
    for i in random.sample(glob.glob('*/faris.*'), 35):
        shutil.move(i, 'valid/faris')        
    for i in random.sample(glob.glob('*/arbaaz.*'), 35):
        shutil.move(i, 'valid/arbaaz')
    for i in random.sample(glob.glob('*/arkam.*'), 35):
        shutil.move(i, 'valid/arkam')        
    for i in random.sample(glob.glob('*/nawazish.*'), 35):
        shutil.move(i, 'valid/nawazish')
    for i in random.sample(glob.glob('*/aaron.*'), 35):
        shutil.move(i, 'valid/aaron')        
    for i in random.sample(glob.glob('*/anthony.*'), 35):
        shutil.move(i, 'valid/anthony')
    for i in random.sample(glob.glob('*/juan.*'), 15):
        shutil.move(i, 'test/juan')      
    for i in random.sample(glob.glob('*/sam.*'), 15):
        shutil.move(i, 'test/sam')
    for i in random.sample(glob.glob('*/faris.*'), 15):
        shutil.move(i, 'test/faris')      
    for i in random.sample(glob.glob('*/arbaaz.*'), 15):
        shutil.move(i, 'test/arbaaz')
    for i in random.sample(glob.glob('*/arkam.*'), 15):
        shutil.move(i, 'test/arkam')      
    for i in random.sample(glob.glob('*/nawazish.*'), 15):
        shutil.move(i, 'test/nawazish')
    for i in random.sample(glob.glob('*/aaron.*'), 15):
        shutil.move(i, 'test/aaron')      
    for i in random.sample(glob.glob('*/anthony.*'), 15):
        shutil.move(i, 'test/anthony')        

os.chdir('../../')
print(os.getcwd())

# Check for available GPUs #

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(device_lib.list_local_devices())
print(tf. __version__)
print(keras. __version__)

# Check internet connection #

def test_connection():
    try:
        socket.create_connection(('Google.com', 80))
        return True
    except OSError:
        return False

print(test_connection())

# Preprocess data with Keras #

train_path = 'FC_IMAGES/train'
valid_path = 'FC_IMAGES/valid'
test_path = 'FC_IMAGES/test'
target = (224, 224, 3)
FC_classes = ['aaron', 'anthony', 'arbaaz', 'arkam', 'faris', 'juan', 'nawazish', 'sam']
train_batches = ImageDataGenerator().flow_from_directory(directory=train_path, target_size=(224,224), classes= FC_classes, batch_size=5)
valid_batches = ImageDataGenerator().flow_from_directory(directory=valid_path, target_size=(224,224), classes= FC_classes, batch_size=5)
test_batches = ImageDataGenerator().flow_from_directory(directory=test_path, target_size=(224,224), classes= FC_classes, batch_size=5)

# Image augmentation #

datagen = ImageDataGenerator(
        rotation_range=20,          # rotation
        width_shift_range=0.2,      # horizontal shift
        height_shift_range=0.2,     # vertical shift
        zoom_range=0.2,             # zoom
        horizontal_flip=False,      # horizontal flip
        brightness_range=[0.2,1.2]) # brightness

# Define training, validation, and test data for models #

train_batches = datagen.flow_from_directory(directory=train_path, target_size=(224,224), classes= FC_classes, batch_size=5)
valid_batches = datagen.flow_from_directory(directory=valid_path, target_size=(224,224), classes= FC_classes, batch_size=5)
test_batches = datagen.flow_from_directory(directory=test_path, target_size=(224,224), classes= FC_classes, batch_size=5)

# Set parameters for training #

EPOCHS = 30
checkpoint_filepath = './FC_IMAGES/checkpoint'

    ## first trial of training with EarlyStopping
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir='./FC_IMAGES/logs'),
]

    ## second trial without EarlyStopping
my_callbacks_1 = [
    tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir='./FC_IMAGES/logs'),
]

STEP_SIZE_TRAIN = train_batches.n//train_batches.batch_size
STEP_SIZE_VALID = valid_batches.n//valid_batches.batch_size

# Construct base models #

    # Base model #1: CNN #1

    ## This model achieved ~ 57.46% prediction accuracy
model_1 = Sequential([
    Dense(units=16, input_shape=(224,224,3), activation='relu'),
    Dense(units=32, activation='relu'),
    Flatten(),
    Dense(units=8, activation='softmax')
    ], name='CNN_1')

    # Base model #2: CNN #2

    ## This model achieved ~ 53.45% prediction accuracy
model_2 = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=8, activation='softmax')
    ], name='CNN_2')

    # Base model #3: CNN #3

    ## This model achieved ~ 54.16% prediction accuracy
model_3 = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same'),
    AveragePooling2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    AveragePooling2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=8, activation='softmax')
    ], name='CNN_3')

    # Base model #4: CNN #4

    ## This model achieved ~ 30.06% prediction accuracy
model_4 = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same'),
    AveragePooling2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    AveragePooling2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=32, activation='relu'),
    Dense(units=16, activation='relu'),
    Dense(units=8, activation='softmax')
    ], name='CNN_4')

# Construct SOTA models #

    # SOTA Model #1: MobileNet

    ## This model achieved ~ 32% prediction accuracy only training the classifier without modifying the model's original weights
    ## Achieved ~ 97.0% prediction accuracy when training the entire model, classifier and original model

mobile = tf.keras.applications.MobileNet(input_shape=(224,224,3), include_top = False, weights='imagenet')
print(mobile.summary())

    # SOTA Model #2: MobileNet V2

    ## This model achieved ~ 13% prediction accuracy only training the classifier without modifying the model's original weights
    ## Achieved ~ 96.0% prediction accuracy when training the entire model, classifier and original model

mobile_v2 = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top = False, weights='imagenet')
print(mobile_v2.summary())

    # SOTA Model #3: DenseNet 121

    ## This model achieved ~ 16% prediction accuracy only training the classifier without modifying the model's original weights
    ## Achieved ~ 97.0% prediction accuracy when training the entire model, classifier and original model

dense_net = tf.keras.applications.DenseNet121(input_shape=(224,224,3), include_top = False, weights='imagenet')
print(dense_net.summary())

    # SOTA Model #4: NasNetMobile

    ## This model achieved ~ 20% prediction accuracy only training the classifier without modifying the model's original weights
    ## Achieved ~ 92.0% prediction accuracy when training the entire model, classifier and original model

nas_net = tf.keras.applications.NASNetMobile(input_shape=(224,224,3), include_top = False, weights='imagenet')
print(nas_net.summary())

    # SOTA Model #5: EfficientNet B0

    ## This model achieved ~ 44% prediction accuracy only training the classifier without modifying the model's original weights
    ## Achieved ~ 87.0% prediction accuracy when training the entire model, classifier and original model

eff_net_b0 = tf.keras.applications.EfficientNetB0(input_shape=(224,224,3), include_top = False, weights='imagenet')
print(eff_net_b0.summary())

    # SOTA Model #6: EfficientNet B1

    ## This model achieved ~ 53% prediction accuracy only training the classifier without modifying the model's original weights
    ## Achieved ~ 95.0% prediction accuracy when training the entire model, classifier and original model

eff_net_b1 = tf.keras.applications.EfficientNetB1(input_shape=(224,224,3), include_top = False, weights='imagenet')
print(eff_net_b1.summary())

    # SOTA Model #7: EfficientNet B2

    ## This model achieved ~ 69% prediction accuracy only training the classifier without modifying the model's original weights
    ## Achieved ~ 91.0% prediction accuracy when training the entire model, classifier and original model

eff_net_b2 = tf.keras.applications.EfficientNetB2(input_shape=(224,224,3), include_top = False, weights='imagenet')
print(eff_net_b2.summary())

    # SOTA Model #8: EfficientNet B3

    ## This model achieved ~ 51% prediction accuracy only training the classifier without modifying the model's original weights
    ## Achieved ~ 91.0% prediction accuracy when training the entire model, classifier and original model

eff_net_b3 = tf.keras.applications.EfficientNetB3(input_shape=(224,224,3), include_top = False, weights='imagenet')
print(eff_net_b3.summary())

# Create arrays containing model names for training, testing and saving#

modnames = ['mobile_v1', 'mobile_v1_1', 'mobile_v1_2', 'mobile_v1_3', 'mobile_v2', 'mobile_v2_1', 'mobile_v2_2', 'mobile_v2_3', 'dense_net', 'dense_net_1', 'dense_net_2', 'dense_net_3',
            'nas_net', 'nas_net_1', 'nas_net_2', 'nas_net_3', 'eff_net_b0', 'eff_net_b0_1', 'eff_net_b0_2', 'eff_net_b0_3', 'eff_net_b1', 'eff_net_b1_1', 'eff_net_b1_2', 'eff_net_b1_3',
            'eff_net_b2', 'eff_net_b2_1', 'eff_net_b2_2', 'eff_net_b2_3', 'eff_net_b3', 'eff_net_b3_1', 'eff_net_b3_2', 'eff_net_b3_3']

modnames_1 = ['MobileNet', 'MobileNet V2', 'DenseNet 121', 'NASNetMobile',
            'EfficientNet B0', 'EfficientNet B1', 'EfficientNet B2', 'EfficientNet B3']
              
models = [mobile, mobile_v2, dense_net, nas_net, eff_net_b0, eff_net_b1, eff_net_b2, eff_net_b3]

histpath = r'C:\Users\jlope\Downloads\Computer Files\Python Projects\history'
histpath_1 = r'C:\Users\jlope\Downloads\Computer Files\Python Projects\history_1'

names = ['CNN_1', 'CNN_2', 'CNN_3', 'CNN_4']

models = [model_1, model_2, model_3, model_4]

# Plot image data with labels #

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(10,10))
    for img, ax in zip(images_arr, axes):
        ax.imshow((img * 255).astype(np.uint8))
    plt.tight_layout()
    plt.show()


# Plot images and evaluate model performance #

def model_test(modelset, val=1000):
    try:        
        iter(modelset)
        print("{} is iterable".format(modelset))
        for mod in modelset:
            quan = val
            count = 0
            for num in range(quan):
                test_imgs, test_labels = next(test_batches)
                print(type(test_imgs[0]), test_imgs[0].shape)
                print(type(test_imgs), test_imgs)
                predictions = mod.predict(x=test_imgs, verbose=0)
                rounded_labels=np.argmax(test_labels, axis=1)
                rounded_predictions = np.argmax(predictions, axis=1)
                evalu = test_score(rounded_labels, rounded_predictions)
                count+= evalu        
                print(rounded_labels, rounded_predictions)
                plotImages(test_imgs)
            print('Model, '+mod.name+' has achieved a '+'%'+str((count/quan)*100)+' prediction performance.')

    except TypeError:
        quan = val
        count = 0
        for num in range(quan):
            test_imgs, test_labels = next(test_batches)
            predictions = modelset.predict(x=test_imgs, verbose=0)
            rounded_labels=np.argmax(test_labels, axis=1)
            rounded_predictions = np.argmax(predictions, axis=1)
            evalu = test_score(rounded_labels, rounded_predictions)
            count+= evalu
            print(rounded_labels, rounded_predictions)
            plotImages(test_imgs)
        print('Model,'+modelset.name+'has achieved a'+'%'+str((count/quan)*100)+'prediction performance.')

# Prediction accuracy calculator #

def test_score(tru, pred):
    count = 0
    total = 5
    for num in range(len(tru)):
        if pred[num] == tru[num]:
            count = count+1
    score = count/total
    return score

# Plot confusion matrix #

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    plt.show()
    return

# Function to fit array of models #
def fit_mod(modelset):
    fin_mods = []
    try:
        
        iter(modelset)
        print("{} is iterable".format(modelset))
        for mod in modelset:
            print(mod.name)
            hist = mod.fit(x=train_batches,
                        verbose = 2,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_batches,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=my_callbacks_1,
                        epochs=EPOCHS)
            print('Success!')
            fin_mods.append(mod)
            ## save training history for each model and store into a newly created json file
            history_dict = hist.history
            new = histpath+'\\'+mod.name+'.json'
            with open(new, 'w') as f:
                print("The json file is created")
                json.dump(history_dict, f)
            print('Success!')
        return fin_mods
      
    
    except TypeError:
        print(modelset.name)
        modelset.fit(x=train_batches,
                verbose = 2,
                steps_per_epoch=STEP_SIZE_TRAIN,
                validation_data=valid_batches,
                validation_steps=STEP_SIZE_VALID,
                callbacks=my_callbacks,
                epochs=EPOCHS)
        print('Success!')
        return modelset

# Attach classifier to SOTA model, comile and fit with image data #

def build_mod(modelset, names):
    ## iterate through the set of models
    for val, mod in enumerate(modelset):
        models = []
        modn = []
        ## attach untrained classifier output
        headModel = mod.output
        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(8, activation="softmax")(headModel)
        top_model_1 = tf.keras.Model(inputs=mod.input, outputs=headModel, name=names[4*val])

        ## freeze the convolutional base model weights only, allow only the classifier weights to be modified
        for layer in mod.layers:
            layer.trainable = False

        top_model_1.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        models.append(top_model_1)
        modn.append(top_model_1.name)

        headModel = mod.output
        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(8, activation="softmax")(headModel)
        top_model_2 = tf.keras.Model(inputs=mod.input, outputs=headModel, name=names[4*val+1])

        ## freeze the first 2/3 convolutional base model weights, allowing only the classifier and last third of the convolutional base model layer weights to be modified
        for layer in mod.layers[0:round(2*len(mod.layers)/3)]:
            layer.trainable = False

        top_model_2.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        models.append(top_model_2)
        modn.append(top_model_2.name)

        headModel = mod.output
        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(8, activation="softmax")(headModel)
        top_model_3 = tf.keras.Model(inputs=mod.input, outputs=headModel, name=names[4*val+2])

        ## freeze the first 1/3 convolutional base model weights, allowing only the classifier and last 2 thirds of the convolutional base model layer weights to be modified
        for layer in mod.layers[0:round(len(mod.layers)/3)]:
            layer.trainable = False

        top_model_3.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        models.append(top_model_3)
        modn.append(top_model_3.name)
        
        headModel = mod.output
        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(8, activation="softmax")(headModel)
        top_model_4 = tf.keras.Model(inputs=mod.input, outputs=headModel, name=names[4*val+3])

        ## train the entire model, allowing every weight in the classifier and the convolutional base model to be modified during training
        for layer in mod.layers:            
            layer.trainable = True

        top_model_4.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        models.append(top_model_4)
        modn.append(top_model_4.name)

        models = fit_mod(models)
        
        save_mod(models, modn)
        
        print('Success!')

# Save models #

def save_mod(models, names=[], name=''):
    count=1
    try:
        iter(models)
        print("{} is iterable".format(models))
        for val, mod in enumerate(models):
            if os.path.isfile(('models/'+names[val]+'.h5')) is False:
                mod.save(('models/'+names[val]+'.h5'))
                print('Success!')
            else:
                mod.save(('models/'+names[val]+'_'+str(count)+'.h5'))
                print('Success!')
                count+=count
                
    except TypeError:
        if name=='':
            if os.path.isfile(('models/'+models.name+'.h5')) is False:
                models.save(('models/'+models.name+'.h5'))
                print('Success!')
            else:
                models.save(('models/'+models.name+'_'+str(count)+'.h5'))
                count+=count
                print('Success!')
        else:
            if os.path.isfile(('models/'+name+'.h5')) is False:
                models.save(('models/'+name+'.h5'))
                print('Success!')
            else:
                models.save(('models/'+name+'_'+str(count)+'.h5'))
                count+=count
                print('Success!')
                

# Load models #

def load_mod(modelnames=[], name=''):
    mods=[]
    print(modelnames)
    print(name)
    if name=='' and modelnames!=[]:
        try:
            iter(modelnames)
            print("{} is iterable".format(modelnames))
            for mod in modelnames:
                print(mod)
                new_mod = load_model('models/'+mod+'.h5')
                mods.append(new_mod)
                print('Success!')
                    
        except TypeError:
            new_mod = load_model('models/'+modelnames+'.h5')
            mods.append(new_mod)
            print('Success!')
        return mods
    else:
        new_mod = load_model('models/'+name+'.h5')
        mods.append(new_mod)
        print('Success!')
        return new_mod
        
    
# Saving a model: Method #1: model.save() #

    ## This saves the architecture of the model, allowing to recreate the model.
    ## Also saves the weights of the model, the training configuration (loss, optimizer) and the state of the optimizer,
    ## allowing to resume training exactly where you left off.
if os.path.isfile('models/model.h5') is False:
    model.save('models/model.h5')


# Saving a model: Method #2: model.save_weights() #

    ## If you only need to save the weights of the model, you can use the model.save_weights() function
if os.path.isfile('models/model_weights.h5') is False:
    model.save_weights('models/model_weights.h5')

# Plot training data #

hist = []
for mod in modnames:
    new = histpath+'\\'+mod+'.json'
    history = json.load(open(new, 'r'))
    hist.append(history)
for j in range(1,3):
    x=0
    y=0
    count = 0
    fig, axs = plt.subplots(2, 2)
    if j==2:
        count = 4
    for i in range(4):
        loss_train_1 = hist[4*i*j]['accuracy']
        loss_val_1 = hist[4*i*j]['val_accuracy']
        loss_train_2 = hist[4*i*j+1]['accuracy']
        loss_val_2 = hist[4*i*j+1]['val_accuracy']
        loss_train_3 = hist[4*i*j+2]['accuracy']
        loss_val_3 = hist[4*i*j+2]['val_accuracy']
        loss_train_4 = hist[4*i*j+3]['accuracy']
        loss_val_4 = hist[4*i*j+3]['val_accuracy']
        
        epochs = range(EPOCHS)
        axs[x,y].plot(epochs, loss_train_1, '#7FFFD4', label='V. #1 Training Accuracy')
        axs[x,y].plot(epochs, loss_val_1, '#458B74', label='V. #1 Validation Accuracy')
        axs[x,y].plot(epochs, loss_train_2, '#FF7F24', label='V. #2 Training Accuracy')
        axs[x,y].plot(epochs, loss_val_2, '#8B4513', label='V. #2 Validation Accuracy')
        axs[x,y].plot(epochs, loss_train_3, '#7FFF00', label='V. #3 Training Accuracy')
        axs[x,y].plot(epochs, loss_val_3, '#458B00', label='V. #3 Validation Accuracy')
        axs[x,y].plot(epochs, loss_train_4, '#BF3EFF', label='V. #4 Training Accuracy')
        axs[x,y].plot(epochs, loss_val_4, '#68228B', label='V. #4 Validation Accuracy')
        axs[x,y].set_title(modnames_1[i+count]+' : Training and Validation Accuracy, without Data Augmentation')
        axs[x,y].set(xlabel='Epochs',ylabel='Accuracy')
        axs[x,y].legend(loc='right', fontsize=8)
        if i==0:
            y=y+1
        if i==1:
            x=x+1
            y=0
        if i==2:
            y=y+1
 
        if os.path.isfile('Python_Figures/NO_DATA_AUG/'+modnames_1[mod]+'_LossEval.png') is False:
            fig.savefig('Python_Figures/NO_DATA_AUG/'+modnames_1[mod]+'_LossEval.png')
    fig.tight_layout()
    plt.show()

# Testing #

new_mod = load_mod(name='mobile_v1_3')
im = cv2.imread(r'.\data\Practice_Images\rat.jpg')
img_resized = cv2.resize(im, (224,224))
prediction = new_mod.predict(x=img_resized, verbose=0)
mods = [new_mod]
model_test(mods, val=3)

print(type(new_mod))

build_mod(models, modnames)
new_mods = load_mod(modnames)
model_test(new_mods)

mods = fit_mod(models)
save_mod(mods, names)
names = ['mobile_v1_3']
new_mods = load_mod(names)
model_test(new_mods, val=2)
