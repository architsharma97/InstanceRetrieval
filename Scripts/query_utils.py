import numpy as np
from sklearn.cluster import KMeans
import math
from collections import Counter

from tree import *



def bestMatches(VTree,des):
    for d in des:
        leaf = getLeaf(VTree, d)
        for image in leaf.topImages:
            imgId = image[1]
            score = image[0]
            if imgId in allVotedImages:
                allVotedImages[imgId] += score
            else:
                allVotedImages[imgId] = score
    votes = [(v, k) for k, v in allVotedImages.iteritems()]
    votes.sort(reverse=True)
    return votes

def getLeaf(tree, descriptor):
    if len(tree.children) != 0:
        index = 0
        minDist = sys.maxint
        for i in range(0, K):
            d = distance(tree.children[i].center, descriptor)
            if d < minDist:
                minDist = d
                index = i
        return getLeaf(tree.children[index], descriptor)
    else:
        return tree

def distance(v1, v2):
    sum = 0
    for i in range(0, len(v1)):
        sum += ((v1[i] - v2[i]) * (v1[i] - v2[i])) # L2 : good
        # sum += abs(v1[i] - v2[i]) # L1 : bad
    return sum


def feature_predict(kmeans_tree, feature, num_layer = 1, cluster_inf = '0'):
	try:
		distance = []
		feature = feature/np.linalg.norm(feature)
		end = False
		for i in range(kmeans_tree.branch_factor):
			if (str(num_layer)+';'+str(cluster_inf)+str(i)) in kmeans_tree.tree_info.keys():
				distance.append(np.dot(np.transpose( kmeans_tree.tree_info[str(num_layer)+';'+str(cluster_inf)+str(i)]),feature))
			elif str(num_layer)+';'+str(cluster_inf)+str(i)+'end' in kmeans_tree.tree_info.keys():
				print("else "+str(num_layer)+';'+str(cluster_inf)+str(i))
				distance.append(np.dot(np.transpose( kmeans_tree.tree_info[str(num_layer)+';'+str(cluster_inf)+str(i)+'end']),feature))
				end = True
		if str(num_layer)+';'+str(cluster_inf)+str(distance.index(max(distance)))+'end' in kmeans_tree.tree_info.keys():
			indices =  kmeans_tree.branch_info[str(num_layer)+';'+str(cluster_inf)+str(distance.index(max(distance)))+'indices'+'end'+'image']
			weight =  kmeans_tree.branch_info[str(num_layer)+';'+str(cluster_inf)+str(distance.index(max(distance)))+'indices'+'end'+'weights']
			return indices,weight

		return feature_predict(kmeans_tree, feature, num_layer+1, cluster_inf+str(distance.index(max(distance))))
	except:
		print("keys",kmeans_tree.tree_info.keys())

def query(kmeans_tree, feature_matrix):
	scores = np.zeros(kmeans_tree.num_files)
	for feature in feature_matrix:
		indices, weight = feature_predict(kmeans_tree, feature)
		for index in indices:
			scores[index]+=weight
	return scores
