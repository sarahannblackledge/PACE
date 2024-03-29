import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from UNET_2D import *
import tensorflow as tf
from metrics_and_loss import *


# Hard-coded variables
seed = 21
batch_size = 30
n_classes = 7

#Specify where images and corresponding masks are stored
train_img_dir = '/Users/sblackledge/Documents/ProKnow_database/axial_data/train/images'
train_mask_dir = '/Users/sblackledge/Documents/ProKnow_database/axial_data/train/' \
                 'labels'
val_img_dir = '/Users/sblackledge/Documents/ProKnow_database/axial_data/validation/images'
val_mask_dir = '/Users/sblackledge/Documents/ProKnow_database/axial_data/validation/labels'

#Sort img and mask dirs to ensure in same order
img_list = os.listdir(os.path.join(train_img_dir, 'img'))
img_list.sort()  #Very important to sort as we want to match images and masks with same number.
msk_list = os.listdir(os.path.join(train_mask_dir, 'img'))
msk_list.sort()
num_images = len(os.listdir(os.path.join(train_img_dir, 'img')))
print("Total number of training images are: ", num_images)


#Define function to perform additional preprocessing after datagen
def preprocess_data(img, mask, num_class):
    #Scale images
    img = img / 255

    #Convert mask to one-hot
    labelencoder = LabelEncoder()
    n, h, w, c = mask.shape
    mask = mask.reshape(-1, 1)
    mask = labelencoder.fit_transform(mask)
    mask = mask.reshape(n, h, w, c)
    mask = to_categorical(mask, num_class)

    return img, mask

#Define the generator
def trainGenerator(train_img_path, train_mask_path, num_class):

    img_data_gen_args = dict(horizontal_flip=True,
                             vertical_flip=False,
                             fill_mode='nearest')

    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)

    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=batch_size,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=batch_size,
        seed=seed)

    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        print(mask.shape)
        img, mask = preprocess_data(img, mask, num_class)
        print(mask.shape)
        yield (img, mask)

train_img_gen = trainGenerator(train_img_dir, train_mask_dir, num_class=n_classes)
val_img_gen = trainGenerator(val_img_dir, val_mask_dir, num_class=n_classes)


x, y = train_img_gen.__next__()
print(np.unique(y))

#Make sure the generator is working and that the images and masks are lined up
for i in range(0,3):
    plt.figure()
    image = x[i, :, :, 0]
    mask = np.argmax(y[i], axis=2)
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.show()

#Define the model metrics
num_train_images = len(os.listdir('/Users/sblackledge/Documents/SE_segmentation/Train/images/img'))
num_val_images = len(os.listdir('/Users/sblackledge/Documents/ProKnow_database/axial_data/validation/labels/img'))
steps_per_epoch = num_train_images//batch_size
val_steps_per_epoch = num_val_images//batch_size

IMG_HEIGHT = x.shape[1]
IMG_WIDTH = x.shape[2]
IMG_CHANNELS = x.shape[3]
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = build_unet(input_shape, n_classes=n_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy', dice_coefficient])
model.summary()

#Run the model
history=model.fit(train_img_gen,
          steps_per_epoch=steps_per_epoch,
          epochs=50,
          verbose=1,
          validation_data=val_img_gen,
          validation_steps=val_steps_per_epoch)
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from UNET_2D import *
import tensorflow as tf
from metrics_and_loss import *


# Hard-coded variables
seed = 21
batch_size = 30
n_classes = 7

#Specify where images and corresponding masks are stored
train_img_dir = '/Users/sblackledge/Documents/ProKnow_database/axial_data/train/images'
train_mask_dir = '/Users/sblackledge/Documents/ProKnow_database/axial_data/train/' \
                 'labels'
val_img_dir = '/Users/sblackledge/Documents/ProKnow_database/axial_data/validation/images'
val_mask_dir = '/Users/sblackledge/Documents/ProKnow_database/axial_data/validation/labels'

#Sort img and mask dirs to ensure in same order
img_list = os.listdir(os.path.join(train_img_dir, 'img'))
img_list.sort()  #Very important to sort as we want to match images and masks with same number.
msk_list = os.listdir(os.path.join(train_mask_dir, 'img'))
msk_list.sort()
num_images = len(os.listdir(os.path.join(train_img_dir, 'img')))
print("Total number of training images are: ", num_images)


#Define function to perform additional preprocessing after datagen
def preprocess_data(img, mask, num_class):
    #Scale images
    img = img / 255

    #Convert mask to one-hot
    labelencoder = LabelEncoder()
    n, h, w, c = mask.shape
    mask = mask.reshape(-1, 1)
    mask = labelencoder.fit_transform(mask)
    mask = mask.reshape(n, h, w, c)
    mask = to_categorical(mask, num_class)

    return img, mask

#Define the generator
def trainGenerator(train_img_path, train_mask_path, num_class):

    img_data_gen_args = dict(horizontal_flip=True,
                             vertical_flip=False,
                             fill_mode='nearest')

    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)

    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=batch_size,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=batch_size,
        seed=seed)

    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        print(mask.shape)
        img, mask = preprocess_data(img, mask, num_class)
        print(mask.shape)
        yield (img, mask)

train_img_gen = trainGenerator(train_img_dir, train_mask_dir, num_class=n_classes)
val_img_gen = trainGenerator(val_img_dir, val_mask_dir, num_class=n_classes)


x, y = train_img_gen.__next__()
print(np.unique(y))

#Make sure the generator is working and that the images and masks are lined up
for i in range(0,3):
    plt.figure()
    image = x[i, :, :, 0]
    mask = np.argmax(y[i], axis=2)
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.show()

#Define the model metrics
num_train_images = len(os.listdir('/Users/sblackledge/Documents/SE_segmentation/Train/images/img'))
num_val_images = len(os.listdir('/Users/sblackledge/Documents/ProKnow_database/axial_data/validation/labels/img'))
steps_per_epoch = num_train_images//batch_size
val_steps_per_epoch = num_val_images//batch_size

IMG_HEIGHT = x.shape[1]
IMG_WIDTH = x.shape[2]
IMG_CHANNELS = x.shape[3]
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = build_unet(input_shape, n_classes=n_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy', dice_coefficient])
model.summary()

#Run the model
history=model.fit(train_img_gen,
          steps_per_epoch=steps_per_epoch,
          epochs=50,
          verbose=1,
          validation_data=val_img_gen,
          validation_steps=val_steps_per_epoch)






