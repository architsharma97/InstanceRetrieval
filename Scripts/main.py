import sys

import numpy as np
import hierarchical_kmeans as hkm
from pkl_utils import save_obj, load_obj

# Argument 1: Train (0), Validation(1) or Test(2)

def process_scores():

if int(sys.argv[1])==0:
	'''
	Train Mode
	Argument 2: Branch Factor
	Argument 3: Maximum Number of Layers
	'''
	print "Loading Data"
	data=np.load("../Models/VW_Train/visual_words_reduced.npy")
	print "Creating Vocabulary Tree"
	vocab_tree=hkm.h_kmeans(data, '../Models/VW_Train/regions_list.txt')
	vocab_tree.cluster()

	print "Saving Vocabulary Tree"
	save_obj(vocab_tree,'../Models/vocab_tree_1.pkl')

elif sys.argv[2]==1:
	'''
	Validation mode
	Argument 2: Location of the Vocabulary Tree (.pkl) constructed
	'''
	validation_data=np.load('../Models/VW_Val/visual_words_reduced.npy')
	vocab_tree=load_obj(sys.argv[2])
	regions_list=open('../Models/VW_Val/regions_list.txt','r').read().splitlines()

	matrix_idx=0
	for i in range(len(regions_list)):
		scores=vocab_tree.query(vocab_tree, validation_data[matrix_idx:matrix_idx+regions_list[i],:])
		matrix_idx+=regions_list[i]
else:
	'''
	Test Mode
	Argument 2: Address of the image
	Argument 3: PCAlayer (.pkl) file
	Argument 3: .pkl file of the trained tree
	''' 
	img_address=sys.argv[2]