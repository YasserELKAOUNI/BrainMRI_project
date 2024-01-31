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
from tensorflow.keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
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



from typing import Callable

def charger_data_et_labels(path_dossier_central, label_manager=None, taille_image=(256, 256), filtre='L'):
    
    categories = os.listdir(path_dossier_central) 
    desired_size=taille_image #taille_image
    images_=[]
    labels_=[]
    data=[]
    for category in categories:
        path_of_label=os.path.join(path,category)
        if os.path.isdir(path_of_label):
            
            for fichier in os.listdir(path_of_label):
                
                path_fichier=os.path.join(path_of_label,fichier)
                
                img = Image.open(os.path.join(path_of_label, fichier))
    
                img = img.resize(desired_size) 
                if img.mode != filtre:
                    img = img.convert(filtre)
    
                img_array = np.array(img)
    
                images_.append(img_array)
                
                if os.path.isfile(path_fichier):
                    label=extraire_label(fichier)
                    labels_.append(label)
                    if label:
                        data.append((img_array,label,fichier))
            
    images = np.array(images_)
    if label_manager :
        labels_=label_manager(labels_)
    return images,labels_


images,labels_=charger_data_et_labels(path,labels_monocalsses)


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
    shuffled_images[:subset_size], shuffled_labels[:subset_size], test_size=0.2
)

# Normalisation 
images_train = images_train / 255.0
images_test = images_test / 255.0




model = tf.keras.Sequential([
       Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
       MaxPooling2D(2, 2),
       Conv2D(64, (3, 3), activation='relu',padding='valid'),
       MaxPooling2D(2, 2),
       Conv2D(128, (3, 3), activation='relu',padding='same'),
       AveragePooling2D(2,2),
       Dropout(0.3),
       Flatten(), 
       Dense(256, activation='relu'),
       Dropout(0.5),
       Dense(2, activation='softmax')]
       )
       
   # Compilation 
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(images_train ,labels_train ,epochs=10,verbose=2, validation_data=(images_test,labels_test))

history=model.history







'''  Graphiques de la fonction de perte et la précision   '''

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




''' Prédiction et Matrice de Confusion : Data de validation '''



predictions = model.predict(images_test)

predicted_labels = np.argmax(predictions, axis=1)

true_labels = np.argmax(labels_test, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(true_labels, predicted_labels)

# Print classification report
print("\nClassification Report:")
print(class_report)










''' Fichier  TESTING '''

''' costruction de la matrice de confusion '''

# extraction des données test du fichier "testing "

path_test_dir ="/Users/elkaouniyasser/Brain_project/Brain_data/Testing"
categories = os.listdir(path_test_dir)  #
data_Testing=[]


desired_size=(256,256)
images_Testing=[]
True_labels=[]
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

            images_Testing.append(img_array)
            
            if os.path.isfile(path_fichier):
                label=extraire_label(fichier)
                True_labels.append(label)
                if label:
                    data_Testing.append((img_array,label,fichier))
        
images_Testing = np.array(images_Testing)

images_Testing=images_Testing/255.0
True_labels=labels_monocalsses(True_labels)

labels_tests_array = np.array(True_labels).reshape(-1, 1)



# encodage  one-hot , vecteur de 2 labels 
encoder = OneHotEncoder(sparse=False)
y_Testing = encoder.fit_transform(labels_tests_array)


# prédiction 


y_pred_prob = model.predict(images_Testing)
y_pred = (y_pred_prob > 0.5).astype(np.float64) 


#Confusion 

import seaborn as sns
import matplotlib.pyplot as plt


conf_matrix = confusion_matrix(y_Testing.argmax(axis=1), y_pred.argmax(axis=1))

plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=True,
    xticklabels=['Predicted 0', 'Predicted 1'],
    yticklabels=['Actual 0', 'Actual 1'],
    linewidths=.5,
    square=True,
    annot_kws={"fontsize": 14},
)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.show()

