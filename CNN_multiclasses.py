import os
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense, Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




from CNN_monoclasses import extraire_label , charger_data_et_labels


path="/Users/elkaouniyasser/Brain_project/Brain_data/Training"
#data[1]
images,labels_=charger_data_et_labels(path)
image = Image.fromarray(images[0])
image.show()


def index_finder(liste, name: str):
    bad_label = []
    for index, item in enumerate(liste):
        if name in item:
            bad_label.append(index)
    return bad_label

indices_to_replace = [397, 417, 748, 775, 838, 864, 933, 954, 1024, 1047]

for index in indices_to_replace:
    labels_[index] = 'pi'

def index_finder_bis(labels, valid_labels):
    bad_label = []
    for index, label in enumerate(labels):
        if label not in valid_labels:
            bad_label.append(index)
    return bad_label



def replace_closest_label(labels, valid_labels):
    corrected_labels = []
    for label in labels:
        if label.endswith('Tr') and label[:-2] in valid_labels:
            corrected_labels.append(label[:-2])
        else:
            corrected_labels.append(label)
    return corrected_labels


valid_labels = ['gl', 'no', 'pi', 'me']

labels_ = replace_closest_label(labels_, valid_labels)

# nouveau labels débarrasées de 'Tr' en suffixe ... 

bad_label = index_finder_bis(labels_, valid_labels)
print(bad_label)  



# on eeffctue un encodage one-hot-encodig

from sklearn.preprocessing import OneHotEncoder

# on applatit en 2D ...
labels_train_array = np.array(labels_).reshape(-1, 1)



# encodage  one-hot , vecteur de 4 labels 
encoder = OneHotEncoder(sparse=False)
one_hot_labels = encoder.fit_transform(labels_train_array)



    

images_train, images_test, labels_train, labels_test = train_test_split(
    images, one_hot_labels, test_size=0.2
)

# Normalisation 
images_train = images_train / 255.0
images_test = images_test / 255.0




model = tf.keras.Sequential([
       Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
       MaxPooling2D(2, 2),
       Conv2D(64, (3, 3), activation='relu',padding='same'),
       MaxPooling2D(2, 2),
       Conv2D(64, (3, 3), activation='relu',padding='same'),
       MaxPooling2D(2, 2),
       Dropout(0.2),
       Flatten(), 
       Dense(256, activation='relu'),
       Dropout(0.5),
       Dense(4, activation='softmax')]
       )
       
   # Compilation 
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(images_train ,labels_train ,epochs=30,verbose=2, validation_data=(images_test,labels_test))

history=model.history







# garphique

plt.plot(history.history['loss'], label='Training Loss')
#plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Courbe de la Loss')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.legend()
plt.grid()
plt.show()





plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='darkgreen', lw=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='limegreen', lw=2)
plt.axhline(y=0.9, label='Précision souhaitée 90%', color='red', linestyle='--', lw=2)
plt.title('Courbe de précision CNN ', fontsize=16)
plt.xlabel('Époques', fontsize=14)
plt.ylabel('Précision', fontsize=14)
plt.legend(frameon=False, fontsize=12)
plt.grid(ls='--')
plt.tight_layout()
plt.show()

