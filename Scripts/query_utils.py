import numpy as np
from sklearn.cluster import KMeans
import math
from collections import Counter

from tree import *

def bestMatches(VTree,des,num_cluster):
    allVotedImages = {}
    for d in des:
        leaf = getLeaf(VTree, d,num_cluster)
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

def getLeaf(tree, descriptor,num_cluster):
    if len(tree.children) != 0:
        index = 0
        minDist = sys.maxint
        for i in range(0, num_cluster):
            d = distance(tree.children[i].center, descriptor)
            if d < minDist:
                minDist = d
                index = i
        return getLeaf(tree.children[index], descriptor,num_cluster)
    else:
        return tree

def distance(v1, v2):
    sum = 0
    for i in range(0, len(v1)):
        sum += ((v1[i] - v2[i]) * (v1[i] - v2[i])) # L2 : good
        # sum += abs(v1[i] - v2[i]) # L1 : bad
    return sum