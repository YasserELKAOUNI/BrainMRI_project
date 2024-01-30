#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:32:26 2024

@author: elkaouniyasser
"""
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


path="/Users/elkaouniyasser/Brain_project/Brain_data/Training"
categories = os.listdir(path)  #
data=[]


def extraire_label(file_name):
    
    parti1 = file_name.split('-')
    label=parti1[1].split('_')[0][:2]
    return label 


def labels_monocalsses(liste_labels):
    multi_labels = ['gl', 'pi', 'me']
    new_labels=[]
    for label in liste_labels:
        if label in multi_labels:
            new_labels.append('yes')
        else:
            new_labels.append(label)
    return new_labels





desired_size=(256,256)
images_=[]
labels_=[]
for category in categories:
    path_of_label=os.path.join(path,category)
    if os.path.isdir(path_of_label):
        
        for fichier in os.listdir(path_of_label):
            
            path_fichier=os.path.join(path_of_label,fichier)
            
            img = Image.open(os.path.join(path_of_label, fichier))

            img = img.resize(desired_size) 
            if img.mode != 'L':
                img = img.convert('L')

            img_array = np.array(img)

            images_.append(img_array)
            
            if os.path.isfile(path_fichier):
                label=extraire_label(fichier)
                labels_.append(label)
                if label:
                    data.append((img_array,label,fichier))
        
images = np.array(images_)


labels_=labels_monocalsses(labels_)

from sklearn.preprocessing import OneHotEncoder

# on applatit en 2D ...
labels_train_array = np.array(labels_).reshape(-1, 1)



# encodage  one-hot , vecteur de 4 labels 
encoder = OneHotEncoder(sparse=False)
one_hot_labels = encoder.fit_transform(labels_train_array)

from sklearn.utils import shuffle

shuffled_images, shuffled_labels = shuffle(images, one_hot_labels)

# Select a subset, e.g., 20% of the data
subset_size = int(len(shuffled_images) * 0.2)

# modélisation 



images_train, images_test, labels_train, labels_test = train_test_split(
    shuffled_images[:subset_size], shuffled_labels[:subset_size], test_size=0.2, random_state=42
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
       Dense(2, activation='softmax')]
       )
       
   # Compilation 
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(images_train ,labels_train ,epochs=10,verbose=2, validation_data=(images_test,labels_test))

history=model.history








# garphique

plt.plot(history.history['loss'], label='Training Loss')
#plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Courbe de la Loss du modèle binaire')
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
