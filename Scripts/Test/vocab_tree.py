import numpy as np
from sklearn.decomposition import PCA

# loading visual words_reduced
visual_words = np.load('../visual_words.npy')

# PCA
pca=PCA(n_components=500)
visual_words_reduced=pca.fit_transform(visual_words)

print "Compute Unstructured Hierarchical Clustering..."
st = time.time()
ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(visual_words_reduced)
elapsed_time = time.time() - st
label = ward.labels_
print "Elapsed time: %.2fs" %(elapsed_time)
print "Number of points: %i" %(label.size)