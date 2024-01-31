
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:33:58 2023

@author: elkaouniyasser
"""


import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
from tqdm import tqdm


"""  Vérification de dimensions  """

# Fonction pour trouver les dimensions des données à partir de directoire

def shape_images(directory):  
    shapes = set() # Ensemble de tuple Pour connaître Les -differentes- dimensions des données 
    c=0
    for filename in os.listdir(directory):
        c+=1 # Compteur du nombre des images Par fichier
        if filename.endswith('.jpg'):  # ou '.jpg', '.jpeg', etc.
            try:
                img = Image.open(os.path.join(directory, filename))
                img_gray = img.convert('L')  # niveaux de gris (Normalement c'est déjà ok mais juste au cas où)
                shapes.add(img_gray.size)  
            except Exception as e:  # Toujours une petite exception si jamais y'a un ptit problème de chargement de données
                print(f"Impossible de charger l'image {filename}: {e}")
    print(c) # Notre compteur d'image , 
    return shapes

# Utilise cette fonction pour charger tes données et obtenir les dimensions
shapes = shape_images('/Users/elkaouniyasser/Desktop/Training/coronal')
print("Les dimensions uniques des images sont :", shapes)



"""  Changement et Redimensionnement"""


# Du coup voici la fonction qui va s'occuper de 
# Chargement et redimensionnement ...

def load_and_resize_images(directory, size=(256, 256)):
    resized_images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):  # Remplace par la bonne extension si nécessaire
        
            img = Image.open(os.path.join(directory, filename))
            img_resized = img.resize(size, Image.ANTIALIAS) 
            img_gray = img_resized.convert('L') 
            img_array = np.array(img_gray).flatten()
            resized_images.append(img_array)
        
    return np.stack(resized_images)   
        # stack Prend en charge la transformation de listes en vect (array)
        # Du coup après le redimensionnement, On a un tableau
        #Où chaque ligne représente Le tableau ,Des valeurs d'une image. 



#print(len(data_sagittal))
#print(len(data_axial))
#print(len(data_coronal))
#print(len(data_all))



common_size = (256, 256)  # La taille d'usage  en général

try:  # On fait comme ça pour pas qu'il y a pas de mauvaises surprises au niveau de la Nomenclature des images ...
    print( "Chargement et re-dimensionnement des données")
    data_all = load_and_resize_images('/Users/elkaouniyasser/Desktop/Training/all_cuts', size=common_size)
    data_sagittal=load_and_resize_images('/Users/elkaouniyasser/Desktop/Training/sagittal', size=common_size)
    data_coronal=load_and_resize_images('/Users/elkaouniyasser/Desktop/Training/coronal', size=common_size)
    data_axial=load_and_resize_images('/Users/elkaouniyasser/Desktop/Training/axial', size=common_size)
    
    # NOTE : Du coup Eva et Aziz , Mettez vos propres Paths si jamais vous voulez voir 
    
    
    print("Nombre d'images chargées et redimensionnées S :", data_sagittal.shape[0])
    print("Nombre d'images chargées et redimensionnées C:", data_coronal.shape[0])
    print("Nombre d'images chargées et redimensionnées A:", data_axial.shape[0])
    print("Nombre d'images chargées et redimensionnées tout (all) :", data_all.shape[0])
    # j'affiche la première image pour vérification
    plt.imshow(data_all[1].reshape(common_size), cmap='gray')
    plt.show()
except Exception as e:
    print(e)  #  Du coup si jamais y'a un problème il affiche une exception
    
    
#data_all=data
#data_axial=data   
#data_sagittal=data
#data_coronal=data
print('ok Commençons la PCA')



"""  L'analyse ACP  Et courbes """




# Ici  j'ai utilisé une fonction ' IncrementalPCA'  Qui  fait le calcul De la PCA par batch 
#  parce que je n'arrivais pas à le faire avec mon ordinateur (5712*2 tableau numpy !) d'un seul coup 
# La version normal prends 21 gb de mémoire !! c'est  quand même plus rapide ...


def calculate_incremental_pca(data, n_components, batch_size):
    
    
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    # Du coup normalise évidemment les données pour obtenir 
    #Une homogénéité sur les moyennes et le variance
    
    
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    # J'ai juste Ajoutées une petite barre sympa , Pour voir le progrès si jamais L'incrémentation  prend trop de temps ....
    for batch in tqdm(range(0, data_normalized.shape[0], batch_size), desc='Computing de la PCA'):
        ipca.partial_fit(data_normalized[batch:batch+batch_size])
    
    return np.cumsum(ipca.explained_variance_ratio_)


# La fonction de traçage des  courbes de PCA , En fonction de nombre des variations expliquées

def plot_pca_curve(ax, data, n_components, title, elbow_point, batch_size):
    
    
    explained_variance = calculate_incremental_pca(data, n_components, batch_size)
    
    
    ax.plot(explained_variance, label='Variance Expliquée Cumulative', color='blue')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Nombre de Composants', fontsize=12)
    ax.set_ylabel('Variance Expliquée Cumulative', fontsize=12)
    
    # On dessine La ligne "du coude" Pour voir combien de Composants il nous faut il nous faut
    if elbow_point:
        ax.axvline(x=elbow_point, color='red', linestyle='--', label='Coude')
        ax.legend(frameon=True, framealpha=0.8, fancybox=True)
    
    ax.grid(True)

fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
fig.suptitle('Variance Expliquée Cumulative pour Différentes Coupes IRM', fontsize=16)


# Du coup on essaye de fixer le nombre de batch de façon à ce qu'il soit
# supérieur au nombre de composants à tracer !!
batch_size = 1500 

# Vous pouvez bien sûr changer la taille du batch`
# Il faut juste garder en tête que (batch> nb de composants) 
n_components = 400 
# n_components_total =2000
# Vous pouvez mettre cette dernière variable dans la dernière courbe ,
# Dans un Traçage séparé j'avais trouvé que Il y a plus de 1000 Composants Exploitable
# Après ça prend  Vraiment beaucoup de temps de computing ...



# J'ai fixé le coude  arbitrairement à 100 Parce que 
# j'ai trouvé Que c'est ce qui rend le mieux visuellement parlant ...

plot_pca_curve(axs[0, 0], data_sagittal, n_components, 'Sagittal', 100, batch_size)
plot_pca_curve(axs[0, 1], data_axial, n_components, 'Axial', 100, batch_size)
plot_pca_curve(axs[1, 0], data_coronal, n_components, 'Coronal', 100, batch_size)
plot_pca_curve(axs[1, 1], data_all, n_components, 'Toutes les Coupes', 100, batch_size)

plt.show()







""" Implémentation de la PCA"""



# Enfin une implémentation, Incrémentale , Procédons battu par batch (En faisant attention à la fin des batch !!)
#Donc à chaque fois on utilise notre objet ipca , Et on procède du coup de manière À faire un premier fit
# Lors du premier match , Et éviter ensuite De refaire les choses j'ai rencontré à la mémoire 
# Et donc on s'aide de la methode   "partial_fit" ... Et on conclut. 


def incremental_fit_transform(ipca, data, batch_size):
   
    
    for batch in tqdm(range(0, data.shape[0], batch_size), desc='Incremental PCA Fit & Transform'):
        
        end_batch = min(batch + batch_size, data.shape[0])
        # Du coup ici , On fait attentionAu fait Que la terminaison d'un batch ne dépasse pas 
        # Les données dont on dispose En taille 
        
        
        
        if batch == 0:
            # Pour le premier lot, nous utilisons 'fit'
            transformed_batch = ipca.fit_transform(data[batch:end_batch])
        else:
            
            # 
            # Pour les lots suivants, nous utilisons 'partial_fit' et 'transform'
            ipca.partial_fit(data[batch:end_batch])
            transformed_batch = ipca.transform(data[batch:end_batch])

        if batch == 0:
            transformed_data = transformed_batch
        else:
            transformed_data = np.vstack((transformed_data, transformed_batch))
    return transformed_data








data= data_sagittal  #data_all 

# On choisit les données qu'on veut projeter (En brut ici)


#Et on refait une normalisation

scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# objet de  IncrementalPCA
n_components = 200
ipca = IncrementalPCA(n_components=n_components)


batch_size = 1200  # Mêmes règle que tout à l'heure (batch > nb components)

data_transformed = incremental_fit_transform(ipca, data_normalized, batch_size)


# Transformation inversée et  projection 

data_inverted = ipca.inverse_transform(data_transformed)



data_original_scale = scaler.inverse_transform(data_inverted)






# Reshape des données en images et affichage


images_original = data_original_scale.reshape(-1, 256, 256)

#Retour à la forme originale Dimensionnée


echantillion_indices = np.random.choice(images_original.shape[0], 5, replace=False)
# On va prendre 5 photos au hasard et essayer de les visualiserOn original et en profitée !


# Figure de la reconstitution des imagesComparaison 
fig, axes = plt.subplots(2, 5, figsize=(15, 6))




for i, idx in enumerate(echantillion_indices):
    
    # Du coup dans la première ligne , Dans chaque colonne on est les photos originales
    ax = axes[0, i]
    ax.imshow(data[idx].reshape(256, 256), cmap='gray')
    ax.set_title(f"Original {idx}")
    ax.axis('off')
    
    # Ici du coup on compare avecLes mêmes photos projetées Avec 200 composants
    ax = axes[1, i]
    ax.imshow(images_original[idx], cmap='gray')
    ax.set_title(f"Reconstruite {idx}") 
    ax.axis('off')

plt.tight_layout()
plt.show()

