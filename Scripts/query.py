import sys
import time

import numpy as np
import cv2

from utils import *
from pkl_utils import *

from selectivesearch import *
'''
Test Mode
Argument 1: Address of the image
Argument 2: PCAlayer (.pkl) file
Argument 3: .pkl file of the trained tree
''' 

print "Loading vgg16"
model=vgg16()

print "Loading Vocabulary Tree"
vocab_tree=load_obj(sys.argv[3])

print "Loading PCA Layer"
pca=load_obj(sys.argv[2])

img_address=sys.argv[1]

print "Reading in image"
img=cv2.imread(img_address)

print "Selecting regions for the image"
img_lbl, regions=selective_search(img, scale=500,sigma=0.9, min_size=10)

processed_regions=np.zeros((len(regions),224,224,3))

print "Preprocessing Images"
for idx,r in enumerate(regions):
	x,y,w,h=r['rect']

	# crops the region and resizes it to 224x224
	crop_img=cv2.resize(img[y:y+h+1,x:x+w+1],(224,224))
	processed_regions[idx,:,:,:]=process_image(crop_img)

print "Getting the features"
feature_matrix=model.predict(processed_regions)
feature_matrix=pca.transform(feature_matrix)

print "Getting the rankings"
score=vocab_tree.query_tree(vocab_tree, feature_matrix)
rankings=sorted(range(len(score)), key=lambda k: score[k])