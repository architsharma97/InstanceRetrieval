import numpy as np

visual_words = np.load('../visual_words.npy')

print "Compute Unstructured Hierarchical Clustering..."
st = time.time()
ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(visual_words)
elapsed_time = time.time() - st
label = ward.labels_
print "Elapsed time: %.2fs" %(elapsed_time)
print "Number of points: %i" %(label.size)

