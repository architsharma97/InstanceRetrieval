import sys
import cv2
import numpy as np
from selectivesearch import *
from utils import *

# Argument 1: takes in the list of training images from the dataset
# Argument 2: scale paramenter for selective search

DIR='../Dataset/'
scale=int(sys.argv[2])

def main():
	print "Opening list of training images"
	train_list=open(sys.argv[1],'r').read().splitlines()
	
	# for every image, a numpy matrix (no. of regions x 4096) will be made and appended
	visual_words=[]

	print "Loading VGG16"
	model=vgg16()

	for name in train_list:
		img=cv2.imread(DIR+name)

		print "Selecting regions for " + name
		img_lbl, regions=selective_search(img,scale=scale,sigma=0.9, min_size=10)
		print "Number of regions: " + len(regions)
		features=np.zeros((len(regions),4096))
		
		print "Computing features for selected regions"
		for idx,r in enumerate(regions):
			x,y,w,h=r['rect']

			# crops the region and resizes it to 224x224
			crop_img=cv2.resize(img[y:y+h+1,x:x+w+1],(224,224))
			features[i,:]=model.predict(process_image(crop_img))

		visual_words.append(features)
if __name__ == '__main__':
	main()