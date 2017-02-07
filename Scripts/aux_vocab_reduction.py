import sys
from pkl_utils import save_obj, load_obj
import numpy as np

# Argument 1: Reduced dimesnsions
dim_red=int(sys.argv[1])

matrix_sizes=[long(size) for size in open('../Models/VW_Train/shape_info.txt','r').read().splitlines()]

# final reduced matrix
visual_words_reduced=np.zeros((matrix_sizes[8],dim_red))

# loading precomputed pca
pca=load_obj('../Models/PCAlayer_128.pkl')

done_till_now=0

# load 2nd, 3rd, 4th, 5th, 6th, 7th and 8th matrix separately and reduce them
for i in range(1,9):
	visual_words=np.load('../Models/VW_Train/visual_words_'+str(i)+'.npy')
	visual_words_reduced[done_till_now:done_till_now+matrix_sizes[i-1],:]=pca.transform(visual_words)
	done_till_now+=matrix_sizes[i-1]

print "Saving the complete reduced matrix"
np.save('../Models/VW_Train/visual_words_reduced.npy',visual_words_reduced)