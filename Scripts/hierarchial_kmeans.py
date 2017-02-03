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

	def __init__(self, data_matrix = None,inv_file = 'dummy_file_list.txt' num_leaves = None):
		"""Initialize the class with data matrix"""

		self.data_matrix = data_matrix
		self.num_leaves = num_leaves
		self.tree = None
		with open(inv_file,'r') as f:
			self.inv_file = f.read().split('\n')[:-1]

	def sub_cluster(self,data_cluster, num_cluster):
		pass 
		

	def cluster(self, branch_factor = 3):
		"""
		Creates a tree with leaves stored in the tree array 
		and labels of each node stored in labels array
		Arguments:
			branch_factor(int): Number of branches at each node
		"""
		self.tree = []
		self.labels = []
		self.branch_factor = branch_factor
		self.weight_matrix = []
		self.node_file = []
		

		num_data_points = self.data_matrix.shape[0]


		num_image = 0
		for each in self.inv_file:
			for i in range(each):
				self.node_file.append(num_image)
			num_image+=1


		num_layers = int(math.log(num_data_points, self.branch_factor))
		num_layer_cluster = num_data_points/self.branch_factor #convert to ceil
		# X = self.data_matrix
		# self.tree.append(X)


		def sub_cluster(self,data_cluster, num_cluster):
			kmeans = KMeans(num_cluster, random_state=0).fit(data_cluster)
			self.tree.extend(kmeans.cluster_centers_)
			labels = kmeans.labels_
			for i in range(num_cluster):
				indices = np.array([j for j, x in enumerate(labels) if x == i])
				new_cluster = data_cluster[indices,:]
				sub_cluster(new_cluster, num_cluster)

		sub_cluster(self.data_matrix,self.branch_factor)

			#weight matrix computation



		# for i in range(num_layers):
			
		# 	self.labels.append(kmeans.labels_)
		# 	for j in range(num_layer_cluster):
		# 		c[i].append(self.labels[i].count())
		# 	X = kmeans.cluster_centers_
		# 	num_layer_cluster = num_layer_cluster/self.branch_factor
		# 	self.tree.append(X)

	def predict(self):
		pass
