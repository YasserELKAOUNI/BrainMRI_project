#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:33:23 2024

@author: elkaouniyasser
"""

import os
from PIL import Image
import numpy as np

''' foctions utiles '''

def extraire_label(file_name):
    
    parti1 = file_name.split('-')
    label=parti1[1].split('_')[0][:2]
    return label 


def labels_monoclasses(liste_labels):
    multi_labels = ['gl', 'pi', 'me']
    new_labels=[]
    for label in liste_labels:
        if label in multi_labels:
            new_labels.append('yes')
        else:
            new_labels.append(label)
    return new_labels




def charger_data_et_labels(path_dossier_central, label_manager=None, taille_image=(256, 256), filtre='L'):
    
    categories = os.listdir(path_dossier_central) 
    desired_size=taille_image #taille_image
    images_=[]
    labels_=[]
    data=[]
    for category in categories:
        path_of_label=os.path.join(path_dossier_central,category)
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









''' fonctions supplémentaires '''

def index_finder(liste, name: str):
    bad_label = []
    for index, item in enumerate(liste):
        if name in item:
            bad_label.append(index)
    return bad_label

#indices_to_replace = [397, 417, 748, 775, 838, 864, 933, 954, 1024, 1047]



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



print("module interne fonctions_preprocessing importé")
