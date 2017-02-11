import sys
import time

import numpy as np
import cv2
from sklearn.externals import joblib

from utils import *
from pkl_utils import *
from hierarchical_kmeans import *
import query_utils

from selectivesearch import *

'''
Test Mode
Argument 1: address of the file containing addresses of the test images
Argument 2: PCAlayer (.pkl) file
Argument 3: .pkl file of the trained tree
Argument 4: Original list of training file names
'''

print ("Loading vgg16")
model=vgg16()

print ("Loading Vocabulary Tree")
vocab_tree   = load_obj(sys.argv[3])
num_clusters = int(sys.argv[3].split('/')[-1].split('_')[2])

print ("Loading PCA Layer")
pca=load_obj(sys.argv[2])

address_file=open(sys.argv[1],'r').read().splitlines()
for address in address_file:
	t1=time.time()
	
	print ("Reading in image")
	img=cv2.imread(address)
	img_address=address.split('/')
	img_name=img_address[len(img_address)-1].split('.')[0]

	print ("Selecting regions for the image")
	img_lbl, regions=selective_search(img, scale=500,sigma=0.9, min_size=10)

	processed_regions=np.zeros((len(regions),224,224,3))

	print ("Preprocessing Images")
	for idx,r in enumerate(regions):
		x,y,w,h=r['rect']

		# crops the region and resizes it to 224x224
		crop_img=cv2.resize(img[y:y+h+1,x:x+w+1],(224,224))
		processed_regions[idx,:,:,:]=process_image(crop_img)

	print ("Getting the features")
	feature_matrix=model.predict(processed_regions)
	feature_matrix=pca.transform(feature_matrix)

	print ("Getting the rankings")
	rankings=query_utils.bestMatches(vocab_tree, feature_matrix, num_clusters)
	print rankings[:10]
	print rankings[len(rankings)-10:]
	train_file_names=open(sys.argv[4],'r').read().splitlines()

	file_rankings=[train_file_names[rank[1]] for rank in rankings]

	print ("Outputting the file")

	output=open('/'.join(img_address[:len(img_address)-1])+'/'+img_name+'.txt','w')

	for name in file_rankings:
		folder, img_code = name.split('/')
		output.write(img_code + ' ' + folder+'\n')

	output.close()

	print ("Time to complete the query: " + str(time.time()-t1) + " seconds")
