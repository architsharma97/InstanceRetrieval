import sys
import os

import numpy as np
from sklearn.externals import joblib

import hierarchical_kmeans as hkm
from pkl_utils import *
# import deepdish as dd
# Argument 1: Train (0), Validation(1) or Test(2)

def process_scores(score):
	rankings=sorted(range(len(score)), key=lambda k: score[k])
	return rankings

if int(sys.argv[1])==0:
	'''
	Train Mode
	Argument 2: Branch Factor
	Argument 3: Maximum Number of Layers
	'''
	print ("Loading Data")
	data=np.load("../Models/VW_Train/visual_words_reduced.npy")
	
	print ("Creating Vocabulary Tree")
	vocab_tree=hkm.h_kmeans(data, '../Models/VW_Train/regions_list.txt')
	vocab_tree.cluster(int(sys.argv[2]), int(sys.argv[3]))

	print ("Saving Vocabulary Tree")
	# try:
	if not os.path.exists('../Models/vocab_tree_'+sys.argv[2]+'_'+sys.argv[3]+'/'):
		os.path.mkdirs('../Models/vocab_tree_'+sys.argv[2]+'_'+sys.argv[3]+'/')
	DIR='../Models/vocab_tree_'+sys.argv[2]+'_'+sys.argv[3]+'/'
	joblib.dump(vocab_tree, DIR+'vocab_tree_'+sys.argv[2]+'_'+sys.argv[3]+'.pkl')
	# except:
		# dd.io.save('../Models/vocab_tree_'+sys.argv[2]+'_'+sys.argv[3]+'.h5',[vocab_tree.inv_file,vocab_tree.num_files,vocab_tree.tree_info,vocab_tree.branch_info,vocab_tree.branch_factor])

elif int(sys.argv[1])==1:
	'''
	Validation mode
	Argument 2: Location of the Vocabulary Tree (.pkl) constructed
	Argument 3: Output file for Validation
	'''
	validation_data=np.load('../Models/VW_Val/visual_words_reduced.npy')
	vocab_tree=load_obj(sys.argv[2])
	regions_list=[int(num) for num in open('../Models/VW_Val/regions_list.txt','r').read().splitlines()]
	output=open(sys.argv[3],'w')

	matrix_idx=0
	for i in range(len(regions_list)):
		score=vocab_tree.query(vocab_tree, validation_data[matrix_idx:matrix_idx+regions_list[i],:])
		rankings=process_scores(score)
		for file_idx in rankings[:len(rankings)-1]:
			output.write(str(file_idx)+',')
		output.write(str(rankings[len(rankings)-1])+'\n')
		matrix_idx+=regions_list[i]
