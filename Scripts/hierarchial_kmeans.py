import numpy as np
from sklearn.cluster import KMeans
import math

class h_kmeans:
	"""
	Arguments:
		data_matrix: NXM Matrix [N:(Sample Size), M:(Features)]
	Methods:
		cluster
		predict
	returns h_kmeans object
	"""

	def __init__(self, data_matrix = None, num_leaves = None):
		"""Initialize the class with data matrix"""

		self.data_matrix = data_matrix
		self.num_leaves = num_leaves
		self.tree = None

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

		num_data_points = self.data_matrix.shape[0]
		num_layers = int(math.log(num_data_points, self.branch_factor))
		num_layer_cluster = num_data_points/self.branch_factor #convert to ceil
		X = self.data_matrix
		self.tree.append(X)

		for i in range(num_layers):
			kmeans = KMeans(num_layer_cluster, random_state=0).fit(X)
			self.labels.append(kmeans.labels_)

			X = kmeans.cluster_centers_
			num_layer_cluster = num_layer_cluster/self.branch_factor
			self.tree.append(X)

	def predict():
		pass