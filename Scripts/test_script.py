from utils import *
import skimage.data
from selectivesearch import *
import cv2
import time

t1=time.time()

# loading vgg16 model
model=vgg16()
t2=time.time()

# selective search
img=cv2.imread("test.jpg")
img_lbl, regions=selective_search(img, scale=5, sigma=0.9, min_size=10)
print len(regions)
t3=time.time()

# feature computation for selected regions
cropped_images=np.zeros((100,224,224,3))
for i,r in enumerate(regions[:100]):
	print r
	x,y,w,h=r['rect']
	crop_img=cv2.resize(img[y:y+h+1,x:x+w+1],(224,224))
	cropped_images[i,:,:,:]=process_image(crop_img)
features=model.predict(cropped_images)
print features.shape
t4=time.time()

print "Time taken to load VGG16: " + str(t2-t1)
print "Time taken for getting regions from Selective Search: " + str(t3-t2)
print "Time taken to get VGG16 features for 100 regions: " + str(t4-t3)
