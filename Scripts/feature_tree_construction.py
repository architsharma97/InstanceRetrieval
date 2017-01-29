import sys
import cv2
import numpy as np
from selectivesearch import *
from utils import *
from sklearn.decomposition import PCA

# Argument 1: takes in the list of training images from the dataset
# Argument 2: scale paramenter for selective search

DIR='../Dataset/'
scale=int(sys.argv[2])

def main():
	print "Opening list of training images"
	train_list=open(sys.argv[1],'r').read().splitlines()
	
	# for every image, a numpy matrix (no. of regionsx224x224x3) will be made and appended
	# added pseudo row which will be removed later
	images=np.zeros((1,224,224,3))

	print "Loading VGG16"
	model=vgg16()

	for name in train_list:
		img=cv2.imread(DIR+name)

		print "Selecting regions for " + name
		img_lbl, regions=selective_search(img,scale=scale,sigma=0.9, min_size=10)
		print "Number of regions: " + str(len(regions))

		# crop and process all images, and then feed to CNN 
		processed_regions=np.zeros((len(regions),224,224,3))
		
		print "Computing features for selected regions"
		for idx,r in enumerate(regions):
			x,y,w,h=r['rect']

			# crops the region and resizes it to 224x224
			crop_img=cv2.resize(img[y:y+h+1,x:x+w+1],(224,224))
			processed_regions[idx,:,:,:]=process_image(crop_img)
			
		images=np.append(images,processed_regions,axis=0)

	# extraction of 4096 dimensional features
	visual_words=model.predict(images[1:,:,:,:])
	
	print "Shape of the visual words matrix: ", 
	print visual_words.shape

	print "Computed all the features: Dimensionality Reduction"
	pca=PCA(n_components=500)
	visual_words_reduced=pca.fit_transform(visual_words)

	print "Saving the Visual Words"
	np.save('../Models/visual_words',visual_words)

if __name__ == '__main__':
	main()