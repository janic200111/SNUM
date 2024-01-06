import os
import random
import numpy as np
from PIL import Image 

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Conv2DTranspose
from tensorflow.keras.models import Model,load_model,Sequential
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Binarizer
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import matplotlib.pyplot as plt
import sys


epochs = 100
PART_SIZE=256
lr =0.001
batches=10000
sactivation = 'sigmoid'
sufix="_1stHO.png"
data_test_ratio=0.3
num_of_rotations=2
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def load_data(path, test_ratio=data_test_ratio, patch_size=(PART_SIZE, PART_SIZE)):
    file_list = os.listdir(path)
    random.shuffle(file_list)
    num_test = int(len(file_list) * test_ratio)

    X_test = []
    Y_test = []
    X_train = []
    Y_train = []

    for i, file_name in enumerate(file_list):
        file_path = os.path.join(path, file_name)
        image = Image.open(file_path)
        image = image.resize((PART_SIZE, PART_SIZE))
        bw_image = image.convert('L')
        bw_image = np.array(bw_image) / 255.0
        bw_image = np.expand_dims(bw_image, axis=-1)
        if i < num_test:
            X_test.append(bw_image)
            Y_test.append(np.array(image)/ 255.0)
        else:
            X_train.append(bw_image)
            Y_train.append(np.array(image)/ 255.0)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_test, Y_test, X_train, Y_train

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Pierwsza warstwa Conv2DTranspose - rozmiar zwiększony do 128x128

    model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))

    # Druga warstwa Conv2DTranspose - przywraca oryginalny rozmiar 256x256
    model.add(Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same'))
    model.add(Activation('sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[ssim_metric])
    return model


if sys.argv[1] == '1':
    X_test, Y_test, X_train, Y_train = load_data('zdjecia_test', patch_size=(PART_SIZE, PART_SIZE))
    model = create_model((PART_SIZE, PART_SIZE,1))
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batches, epochs=epochs)

    model.save('model_1.h5')

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
else:
    model = load_model('model_1.h5')

obraz = load_img('zdjecia/20220805_214821.jpg', target_size=(256, 256), color_mode='grayscale')
bw_image = obraz.convert('L')
obraz = img_to_array(obraz)
obraz_1 =obraz
obraz = obraz / 255.0  # Normalizacja
obraz = np.expand_dims(obraz, axis=0)
wynik = model.predict(obraz)
wynikowy_obraz = wynik[0]
wynikowy_obraz = (wynikowy_obraz * 255).astype(np.uint8)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 wiersz, 2 kolumny, rozmiar figury 12x6 cali

# Wyświetlanie pierwszego obrazu
axes[0].imshow(wynikowy_obraz)
axes[0].set_title('Wynikowy Obraz')
axes[0].axis('off')  # Wyłączenie osi dla czytelności

# Wyświetlanie drugiego obrazu
axes[1].imshow(obraz_1)
axes[1].set_title('Obraz 1')
axes[1].axis('off')  # Wyłączenie osi dla czytelności

axes[1].imshow(bw_image)
axes[1].set_title('czarno-bialy')
axes[1].axis('off')  # Wyłączenie osi dla czytelności

# Wyświetlenie wykresu
plt.show()

"""
path1="M:\Dokumnety arch\Studia\IWM\IWM-oko\Images\Image_05L.jpg"
path2="M:\Dokumnety arch\Studia\IWM\IWM-oko\Images\Image_05L_2ndHO.png"

image1 = Image.open(path1)
image2 = Image.open(path2)

plt.subplot(2, 2, 1)
plt.imshow(image1)
plt.title('Orginalny obraz')

plt.subplot(2, 2, 2)
plt.imshow(image2, cmap='gray')
plt.title('Maska')

predict_from_model(path1, model, 0.5,path2)


p=filrt(path1,path2)
plt.subplot(2, 2, 4)
plt.imshow(p, cmap='gray')
plt.title('Wynik filtru')

plt.show()
"""