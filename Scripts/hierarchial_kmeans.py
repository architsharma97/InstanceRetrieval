import numpy as np
from sklearn.cluster import KMeans
import math
from collections import Counter

class h_kmeans:
	"""
	Arguments:
		data_matrix: NXM Matrix [N:(Sample Size), M:(Features)]
	Methods:
		cluster
		predict
	returns h_kmeans object
	"""

	def __init__(self, data_matrix,inv_file = '../dummy_file_list.txt',num_leaves = 5):
		"""Initialize the class with data matrix"""

		self.data_matrix = data_matrix
		self.num_leaves = num_leaves
		self.tree = None
		with open(inv_file,'r') as f:
			self.inv_file = f.read().split('\n')[:-1]
		

	def cluster(self, branch_factor = 3):
		"""
		Arguments:
			branch_factor(int): Number of branches at each node
		"""
		self.tree = []
		self.tree_info = {}
		self.branch_info = {}
		self.labels = []
		self.branch_factor = branch_factor
		self.weight_matrix = []
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
			if data_cluster.shape[0]<self.num_leaves:
				return
			kmeans = KMeans(num_cluster, random_state=0).fit(data_cluster)
			self.tree.extend(kmeans.cluster_centers_.tolist())
			labels = kmeans.labels_
			
			for i in range(num_cluster):
				indices = np.array([j for j, x in enumerate(labels) if x == i])
				self.tree_info[str(num_layer)+';'+cluster_inf+str(i)] = kmeans.cluster_centers_[i,:]
				self.branch_info[str(num_layer)+';'+cluster_inf+str(i)+'indices'] = indices
				new_cluster = data_cluster[indices,:]
				sub_cluster(new_cluster, num_cluster, num_layer,cluster_inf+str(i))

		sub_cluster(self.data_matrix,self.branch_factor,0, '0')


	def predict(self):
		pass
