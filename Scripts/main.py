import sys

import numpy as np
import hierarchical_kmeans as hkm
from pkl_utils import save_obj

# Argument 1: Train (0), Validation(1) or Test(2)

if int(sys.argv[1])==0:
	'''
	Train Mode
	Argument 2: Branch Factor
	Argument 3: Maximum Number of Layers
	'''
	data=np.load("../Models/VW_Train/visual_words_reduced.npy")
	vocab_tree=hkm.h_kmeans(data, '../Models/VW_Train/regions_list.txt')
	vocab_tree.cluster()
	save_obj(vocab_tree,'../Models/vocab_tree_1.pkl')

# elif sys.argv[2]==1:

else:
	'''
	Test Mode
	Argument 2: Address of the image
	Argument 3: .pkl file of the trained tree
	''' 
	img_address=sys.argv[2]