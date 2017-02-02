import sys
import time
from random import shuffle

import cv2
import numpy as np

from selectivesearch import *
from utils import *

from sklearn.decomposition import PCA

# Argument 1: takes in the list of training images from the dataset
# Argument 2: scale paramenter for selective search

DIR='../Dataset/train/'
scale=int(sys.argv[2])

print "Loading VGG16"
model=vgg16()

def get_visual_words(file_idx, train_list, regions_list):
	'''
	Extraction of visual words is split. The results hereby will be approximate.
	idx: Portion of actual training data (1<=idx<=8)
	train_list: directory of images
	scale: scale at which selective search will operate
	'''
	# print "Opening list of training images"
	# train_list=open(sys.argv[1],'r').read().splitlines()
	
	# a list is maintained which contains numpy matrices.
	regions_by_image=[]
	
	t2=time.time()
	for name in train_list:
		img=cv2.imread(DIR+name)

		print "Selecting regions for " + name
		img_lbl, regions=selective_search(img, scale=scale,sigma=0.9, min_size=10)
		print "Number of regions: " + str(len(regions))
		regions_list.write(str(len(regions))+"\n")
		# crop and process all images, and then feed to CNN 
		processed_regions=np.zeros((len(regions),224,224,3))
		
		print "Computing features for selected regions"
		for idx,r in enumerate(regions):
			x,y,w,h=r['rect']

			# crops the region and resizes it to 224x224
			crop_img=cv2.resize(img[y:y+h+1,x:x+w+1],(224,224))
			processed_regions[idx,:,:,:]=process_image(crop_img)

		# extraction of 4096 dimensional features: appended into a list
		regions_by_image+=[model.predict(processed_regions)]

	t3=time.time()

	tot_regions=0
	for regions in regions_by_image:
		tot_regions+=regions.shape[0]

	visual_words=np.zeros((tot_regions, 4096))
	
	# maintains count of regions donw
	regions_count=0
	for regions in regions_by_image:
		visual_words[regions_count:regions_count+regions.shape[0],:]=regions
		regions_count+=regions.shape[0]

	t4=time.time()
	print "Shape of the visual words matrix: ", 
	print visual_words.shape

	# print "Computed all the features: Dimensionality Reduction"
	# pca=PCA(n_components=500)
	# visual_words_reduced=pca.fit_transform(visual_words)

	# t5=time.time()
	print "Saving the full sized visual Words"
	np.save('../Models/visual_words_'+str(file_idx)+'.npy', visual_words)

	print "Completed computation of part " + str(file_idx) + " vocabulary space"
	print "Times required "
	print "Feature Extraction + Selecting Regions of Interest: %.2fs" %(t3-t2)
	print "Concatatenation of Regions: %.2fs" %(t4-t3)
	# print "PCA: %.2fs" %(t5-t4)

train_list=open(sys.argv[1],'r').read().splitlines()
# shuffle(train_list)
n_images=len(train_list)
regions_list=open("../Models/regions_lixt.txt",'w')
for idx in range(1,9):
	get_visual_words(idx, train_list[n_images*(idx-1)/8:n_images*idx/8], regions_list)
