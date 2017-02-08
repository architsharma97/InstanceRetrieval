import numpy as np
from sklearn.cluster import KMeans
import math
from collections import Counter

def feature_predict(kmeans_tree,feature,num_layer = 1, cluster_inf = '0'):
		distance = []
		feature = feature/ np.linalg.norm(feature)
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

class h_kmeans:
	"""
	Arguments:
		data_matrix: NXM Matrix [N:(Sample Size), M:(Features)]
	Methods:
		cluster
		predict
	returns h_kmeans object
	"""

	def __init__(self, data_matrix,inv_file = '../dummy_file_list.txt'):
		"""Initialize the class with data matrix"""

		self.data_matrix = data_matrix
		# self.num_leaves = num_leaves
		self.tree = None
		with open(inv_file,'r') as f:
			self.inv_file = f.read().split('\n')[:-1]
			self.num_files = len(self.inv_file)
		

	def cluster(self, branch_factor = 3,max_layer = 3):
		"""
		Arguments:
			branch_factor(int): Number of branches at each node
		"""
		self.tree = []
		self.tree_info = {}
		self.branch_info = {}
		self.labels = []
		self.branch_factor = branch_factor
		
		self.node_file = []
		

		num_data_points = self.data_matrix.shape[0]


		num_image = 0
		for each in self.inv_file:
			for i in range(int(each)):
				self.node_file.append(num_image)
			num_image+=1
		
		def sub_cluster(data_cluster, num_cluster, num_layer, cluster_inf):
			num_layer+=1
			print(data_cluster.shape)
			if data_cluster.shape[0]<num_cluster or num_layer>max_layer:
				return
			kmeans = KMeans(num_cluster, random_state=0).fit(data_cluster)
			self.tree.extend(kmeans.cluster_centers_.tolist())
			labels = kmeans.labels_
			
			for i in range(num_cluster):
				indices = np.array([j for j, x in enumerate(labels) if x == i])
				
				new_cluster = data_cluster[indices,:]

				if new_cluster.shape[0]<num_cluster or num_layer>max_layer:
					norm = kmeans.cluster_centers_[i,:] / np.linalg.norm(kmeans.cluster_centers_[i,:])
					self.tree_info[str(num_layer)+';'+cluster_inf+str(i)+'end'] = norm
					self.branch_info[str(num_layer)+';'+cluster_inf+str(i)+'indices'+'end'] = indices
				else:
					norm = kmeans.cluster_centers_[i,:] / np.linalg.norm(kmeans.cluster_centers_[i,:])
					self.tree_info[str(num_layer)+';'+cluster_inf+str(i)] = norm
					self.branch_info[str(num_layer)+';'+cluster_inf+str(i)+'indices'] = indices

				sub_cluster(new_cluster, num_cluster, num_layer,cluster_inf+str(i))

		sub_cluster(self.data_matrix,self.branch_factor,0, '0')


		for key in self.branch_info.keys():
			if 'end' in key:
				node_indices = self.branch_info[key]
				image_indices = [self.node_file[x] for x in node_indices]
				image_indices = list(set(image_indices))
				self.branch_info[key+'image'] = image_indices
				self.branch_info[key+'weights'] = math.log(float(self.num_files)/ len(image_indices))

	def query(kmeans_tree, feature_matrix):
		scores = np.zeros(kmeans_tree.num_files)
		for feature in feature_matrix:
			indices, weight = feature_predict(kmeans_tree, feature)
			for index in indices:
				scores[index]+=weight
		return scores