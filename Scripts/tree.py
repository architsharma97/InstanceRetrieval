
import os
import math
# import pickle
import sys
import time

from sklearn.cluster import KMeans


class VocabTree:

	def __init__(self):
		self.center   = None
		self.children = []
		# only for leaf nodes:
		self.descriptors = []
		self.images = []
		self.scores = []
		self.imageIndices = []
		self.topImages = []

def generateVocabTree(descriptors, level, num_clusters, L):
	t = time.time()
	vtree = VocabTree()
	print(str(t))
	if level == 0:
		vtree.descriptors = descriptors
		return vtree
	try:
		km = KMeans(n_clusters=num_clusters)
		ds = [d[1] for d in descriptors]
		clusters = km.fit(ds)
		clusterCenters = clusters.cluster_centers_
		C = []
		for i in range(0, num_clusters):
			C.append([])
		for d in descriptors:
			C[clusters.predict(d[1].reshape(1,-1))[0]].append(d)
		for i in range (0, num_clusters):
			vtree.children.append(generateVocabTree(C[i], level-1,num_clusters,L))
		for i in range (0, num_clusters):
			vtree.children[i].center = clusterCenters[i]
	except Exception, e:
		vtree.descriptors = descriptors
		print "K Means exception"
		print e
	if level == L:
		print "VTree complete in " + str(time.time() -t) + " s"
	return vtree

def computeIFIndex(tree,ND):
	t = time.time()
	
	if len(tree.children) != 0:
		for child in tree.children:
			computeIFIndex(child,ND)
	else:
		Ni = len(tree.images)
		for img in tree.images:
			ndi = tree.imageIndices.count(img)
			nd  = ND[img]
			tree.scores.append(float(ndi)/nd*math.log(len(ND)/Ni))
	if tree.center == None:
		print "IF Index complete in " + str(time.time() -t) + " s"

def computeNDArray(tree, ND):
	t = time.time()
	
	if len(tree.children) != 0:
		for child in tree.children:
			computeNDArray(child, ND)
	else:
		tree.imageIndices  = [d[0] for d in tree.descriptors]
		tree.images      = list(set(tree.imageIndices))
		for img in tree.images:
			ND[img] += 1
	if tree.center == None:
		print "ND Array complete in " + str(time.time() -t) + " s"
	return ND

def computeTopImages(tree,ND):
	t = time.time()

	if len(tree.children) != 0:
		for child in tree.children:
			computeTopImages(child,ND)
	else:
		temp = [(tree.scores[i], tree.images[i]) for i in range (0, len(tree.images))]
		temp.sort(reverse=True)
		tree.topImages = temp[:383]
	if tree.center == None:
		print "Top Images complete in " + str(time.time() -t) + " s"
