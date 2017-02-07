# script reads in original feature matrices and reduces the matrix to R^D matrix
import sys
sys.path.append('../')
import time

from pkl_utils import save_obj

from sklearn.decomposition import PCA
import numpy as np

# Argument 1: Number of dimension to which the data need to be reduced
dim_red=int(sys.argv[1])
print "Reducing data to " + str(dim_red) + " dimensions"

matrix_sizes=[long(size) for size in open('../Models/VW_Train/shape_info.txt','r').read().splitlines()]

# approximate PCA
num_words=0
for i in range(6):
	num_words+=matrix_sizes[i]

# matrix to be reduced
feature_matrix=np.zeros((num_words,4096))

# final reduced matrix
visual_words_reduced=np.zeros((matrix_sizes[8],dim_red))

done_till_now=0
t1=time.time()
for i in range(1,7):
	print "Loading " + str(i) + "th visual words matrix for PCA"
	feature_matrix[done_till_now:done_till_now+matrix_sizes[i-1],:]=np.load('../Models/VW_Train/visual_words_'+str(i)+'.npy')
	done_till_now+=matrix_sizes[i-1]

t2=time.time()
print "Performing Dimensionality reduction"
pca=PCA(n_components=dim_red)

print "Reducing and copying the initial 6 matrices"
visual_words_reduced[:feature_matrix.shape[0],:]=pca.fit_transform(feature_matrix)

t3=time.time()
print "Saving PCA matrix in a pickle object"
save_obj(pca, '../Models/PCAlayer_'+str(dim_red)+'.pkl')

t4=time.time()
print "Loading the rest of matrices and reducing them"

# load 7th and 8th matrix separately and reduce them
for i in range(7,9):
	visual_words=np.load('../Models/VW_Train/visual_words_'+str(i)+'.npy')
	visual_words_reduced[done_till_now:done_till_now+matrix_sizes[i-1],:]=pca.tranform(visual_words)
	done_till_now+=matrix_sizes[i-1]

t5=time.time()
print "Saving the complete reduced matrix"
np.save('../Models/VW_Train/visual_words_reduced.npy',visual_words_reduced)
t6=time.time()

print "Time to load first 6 visual words matrices: " + str(t2-t1)
print "Time to perform PCA: " + str(t3-t2)
print "Time to save the PCA object: " + str(t4-t3)
print "Time to load and reduce the rest of matrices: " + str(t5-t4)
print "Time to save the complete reduced matrix: " + str(t6-t5)