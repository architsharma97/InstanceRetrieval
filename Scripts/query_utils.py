import numpy as np
from sklearn.cluster import KMeans
import math
from collections import Counter



def feature_predict(kmeans_tree, feature, num_layer = 1, cluster_inf = '0'):
		distance = []
		feature = feature/np.linalg.norm(feature)
		end = False
		for i in range(kmeans_tree.branch_factor):
			if (str(num_layer)+';'+str(cluster_inf)+str(i)) in kmeans_tree.tree_info.keys():
				distance.append(np.dot(np.transpose( kmeans_tree.tree_info[str(num_layer)+';'+str(cluster_inf)+str(i)]),feature))
			else:
				print("else "+str(num_layer)+';'+str(cluster_inf)+str(i))
				distance.append(np.dot(np.transpose( kmeans_tree.tree_info[str(num_layer)+';'+str(cluster_inf)+str(i)+'end']),feature))
				end = True
		if str(num_layer)+';'+str(cluster_inf)+str(distance.index(max(distance)))+'end' in kmeans_tree.tree_info.keys():
			indices =  kmeans_tree.branch_info[str(num_layer)+';'+str(cluster_inf)+str(distance.index(max(distance)))+'indices'+'end'+'image']
			weight =  kmeans_tree.branch_info[str(num_layer)+';'+str(cluster_inf)+str(distance.index(max(distance)))+'indices'+'end'+'weights']
			return indices,weight

		return feature_predict(kmeans_tree, feature, num_layer+1, cluster_inf+str(distance.index(max(distance))))


def query(kmeans_tree, feature_matrix):
		scores = np.zeros(kmeans_tree.num_files)
		for feature in feature_matrix:
			indices, weight = feature_predict(kmeans_tree, feature)
			for index in indices:
				scores[index]+=weight
		return scores