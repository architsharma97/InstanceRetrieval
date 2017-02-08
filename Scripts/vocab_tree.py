import cv2
import os
import math
# import pickle
import sys
import time

from sklearn.cluster import KMeans



class VocabTree:
    def __init__(self,inv_file = '../dummy_file_list.txt'):
        self.cluster_center   = None
        self.children = []
        # only for terminal
        self.data_cluster = []
        self.images = []
        self.scores = []
        self.indices = []
        self.top_images = []

        with open(inv_file,'r') as f:
			self.inv_file = f.read().split('\n')[:-1]

		

    def build_tree(self,data_cluster, leaf, num_cluster):
    	vocab_tree = VocabTree()

	    if leaf > data_cluster.shape[0]:
	        vocab_tree.data_cluster = data_cluster
			return vocab_tree
		else:
			kmeans = KMeans(num_cluster, random_state=0).fit(data_cluster)
			centers = kmeans.cluster_centers_
			clusters = {}
			for i in range(num_cluster):
				clusters[str(i)] = []
			for point in data_cluster:
				clusters[str(kmeans.predict(point)[0])].append(point)
			for in range(num_cluster):
				vocab_tree.children.append(build_tree(C[str(i)], leaf-1,num_cluster))
			for i in range(num_cluster):
				vocab_tree.children[i].cluster_center = centers[i]
			return vocab_tree

	def computeNDArray(self, tree):
		t = time.time()
		global ND
		if len(tree.children) != 0:
			for child in tree.children:
				computeNDArray(child)
		else:
			tree.imageIndices  = [d[0] for d in tree.descriptors]
			tree.images      = list(set(tree.imageIndices))
		for img in tree.images:
		ND[img] += 1
		if tree.center == None:
		print "ND Array complete in " + str(time.time() -t) + " s"

