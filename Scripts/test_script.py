from utils import *
import skimage.data
from selectivesearch import *
import cv2
import time

t1=time.time()
model=vgg16()
t2=time.time()
img=cv2.imread("test.jpg")
img_lbl, regions=selective_search(img, scale=500, sigma=0.9, min_size=10)
print len(regions)
t3=time.time()
features=np.zeros((10,4096))
for i,r in enumerate(regions[:10]):
	print r
	x,y,w,h=r['rect']
	crop_img=cv2.resize(img[y:y+h+1,x:x+w+1],(224,224))
	features[i,:]=model.predict(process_image(crop_img))
t4=time.time()
print "Time taken to load VGG16: "+str(t2-t1)
print "Time taken for getting regions from Selective Search: "+str(t3-t2)
print "Time taken to get VGG16 features for 10 regions: "+str(t4-t3)
