# coding=utf-8

#######################################################
# LOAD DATA
#######################################################

import os,glob
import numpy
import fitsio
from matplotlib import pyplot as plt




# Initialisation d'une liste pour contenir les images
# et d'une liste pour contenir les types morphologiques
raw_data =[]
type_morph =[]

 

# Définir les chemins vers les images et le catalogue,
mydataDir ='../data5/data/'
mycatalog_path =os.path.join(mydataDir,'data-COSMOS-10000-id.txt')
mypath_template ='../data5/data/image/*'
 

# Chargement des données qui nous intéressent dans le catalogue
ids, mod = numpy.loadtxt(mycatalog_path, unpack=True, usecols=(0,3))

# Chargement des images
for one in glob.glob(mypath_template):
   # Extraction de l'id à partir du nom de fichier
   print(one)
   idi =int(one.split('/')[-1].split('_')[0])
   print(idi)
   modi = mod[ids==idi][0]
   print(modi)
   # On va ignorer les out layer ie les mod == 0
   if modi>0:
      # Ajout de l'image
      data = fitsio.read(one)
      raw_data.append(data)
      # Ajout du type morphologique
      type_morph.append(modi)

# Reformatage en numpy pour plus de facilité
raw_data = numpy.asarray(raw_data)
type_morph = numpy.asarray(type_morph)

#######################################################
# display
#######################################################

# Un petit graphique pour illustrer
fig = plt.figure(figsize=(7,7))
for i in range(9):
   plt.subplot(330+i+1)
   plt.imshow(raw_data[i])
   plt.title('type %d'%(type_morph[i]))
   plt.axis('off')
# Visualiser le plot
plt.show()


#######################################################
# DATA TRANSFORMATION
#######################################################

# Normalisation des images
data_scaled = numpy.asarray([(img-img.mean())/img.std()for img in raw_data])

# Transformation en 1d array
data_1d = data_scaled.reshape((data_scaled.shape[0],-1))

# Vérifions avec la dimension des données sous forme de vecteurs
print(data_1d.shape)

# Et une valeur moyenne d'environ 0
print(data_1d.mean())

