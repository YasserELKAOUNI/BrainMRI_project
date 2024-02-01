#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: elkaouniyasser
"""
import os
from PIL import Image
import numpy as np


import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt





from tensorflow.keras.models import load_model

import fonctions_preprocessing as pre

from fonctions_preprocessing  import extraire_label , charger_data_et_labels,labels_monoclasses

''' Chargement du modèle à tester  '''

model_path="/Users/elkaouniyasser/Brain_project/maincode/my_best_model.epoch07-loss0.14.hdf5"

modèle = load_model(model_path)




''' Fichier  TESTING '''

''' costruction de la matrice de confusion '''

# extraction des données test du fichier "testing "


path_test_dir ="/Users/elkaouniyasser/Brain_project/Brain_data/Testing"


print("Attention :le working dir ici est :",os.getcwd())
print("Attention : l'adresse du fichier training ici est :",path_test_dir)


images_Testing,True_labels=charger_data_et_labels(path_test_dir,labels_monoclasses)


# Normalisation 

images_Testing=images_Testing/255.0
True_labels=labels_monoclasses(True_labels)

labels_tests_array = np.array(True_labels).reshape(-1, 1)



# encodage  one-hot , vecteur de 2 labels 
encoder = OneHotEncoder(sparse=False)
y_Testing = encoder.fit_transform(labels_tests_array)


# prédiction 


y_pred_prob = modèle.predict(images_Testing)
y_pred = (y_pred_prob > 0.5).astype(np.float64) 


#Confusion 

import seaborn as sns


conf_matrix = confusion_matrix(y_Testing.argmax(axis=1), y_pred.argmax(axis=1))







print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_Testing.argmax(axis=1), y_pred.argmax(axis=1))

# Print classification report
print("\nClassification Report:")
print(class_report)




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
