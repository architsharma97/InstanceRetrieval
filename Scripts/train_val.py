import sys
import os

import numpy as np
from sklearn.externals import joblib

from pkl_utils import *
from tree import *
import query_utils

# import deepdish as dd
# Argument 1: Train (0), Validation(1) or Test(2)

# def process_scores(score):
#	rankings=sorted(range(len(score)), key=lambda k: score[k])
#	return rankings

if int(sys.argv[1])==0:
	'''
	Train Mode
	Argument 2: Branch Factor
	Argument 3: Maximum Number of Layers
	'''
	print ("Loading Data")
	data=np.load("../Models/VW_Train/visual_words_reduced.npy")
	descriptors = []

	inv_file=[int(num) for num in open('../Models/VW_Train/regions_list.txt','r').read().splitlines()] 
	num_files = len(inv_file)
	
	num_image = 0
	feature_idx=0
	for each in inv_file:
		for i in range(int(each)):
			descriptors.append([num_image,data[feature_idx,:]])
			feature_idx+=1
		num_image+=1	
	
	print ("Creating Vocabulary Tree")
	# vocab_tree=hkm.h_kmeans(data, '../Models/VW_Train/regions_list.txt')
	# vocab_tree.cluster(int(sys.argv[2]), int(sys.argv[3]))

	vocab_tree = generateVocabTree(descriptors,int(sys.argv[3]),int(sys.argv[2]),int(sys.argv[3]))
	print ("Vocabulary Tree Generated")
	
	N = num_files
	ND = [0] * N
	ND = computeNDArray(vocab_tree, ND)
	print ("ND Array computed")
	
	computeIFIndex(vocab_tree, ND)
	print ("IDF computed")
	
	computeTopImages(vocab_tree, ND)
	print ("Top Images computed")

	print ("Saving Vocabulary Tree")
	# try:
	if not os.path.exists('../Models/vocab_tree_'+sys.argv[2]+'_'+sys.argv[3]+'/'):
		os.mkdir('../Models/vocab_tree_'+sys.argv[2]+'_'+sys.argv[3]+'/')
	DIR='../Models/vocab_tree_'+sys.argv[2]+'_'+sys.argv[3]+'/'
	
	save_obj(vocab_tree,DIR+'vocab_tree_'+sys.argv[2]+'_'+sys.argv[3]+'.pkl')
	
	# joblib.dump(vocab_tree, DIR+'vocab_tree_'+sys.argv[2]+'_'+sys.argv[3]+'.pkl')
	# except:
		# dd.io.save('../Models/vocab_tree_'+sys.argv[2]+'_'+sys.argv[3]+'.h5',[vocab_tree.inv_file,vocab_tree.num_files,vocab_tree.tree_info,vocab_tree.branch_info,vocab_tree.branch_factor])

elif int(sys.argv[1])==1:
	'''
	Validation mode
	Argument 2: Location of the Vocabulary Tree (.pkl) constructed
	Argument 3: Output file for Validation
	'''
	print ("Loading Data")
	validation_data=np.load('../Models/VW_Val/visual_words_reduced.npy')
	print ("Loading Vocabulary Tree")
	vocab_tree=load_obj(sys.argv[2])
	num_clusters = int(sys.argv[2].split('/')[-1].split('_')[2])
	
	regions_list=[int(num) for num in open('../Models/VW_Val/regions_list.txt','r').read().splitlines()]
	output=open(sys.argv[3],'w')

	matrix_idx=0
	for i in range(len(regions_list)):
		print ("Retrieving rankings for images")
		rankings=query_utils.bestMatches(vocab_tree, validation_data[matrix_idx:matrix_idx+regions_list[i],:], num_clusters)
		
		for file_idx in rankings[:len(rankings)-1]:
			output.write(str(file_idx[1])+',')
		output.write(str(rankings[len(rankings)-1][1])+'\n')
		matrix_idx+=regions_list[i]
