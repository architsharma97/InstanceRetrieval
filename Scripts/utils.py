# utility file
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np

# use model.predict(process_image(img_path)) to get the fc2 layer output
def vgg16():
	'''
	Returns the VGG16 model which outputs the 4096 dimensional embeddings
	of images. To be used with process_image.
	'''
	print "Building VGG16 model"
	base_model=VGG16(weights='imagenet',include_top=True)
	model=Model(input=base_model.input, output=base_model.get_layer('fc2').output)
	return model

def process_image(img_path):
	'''
	img_path: path to the actual image
	'''
	print "Pre-processing image"
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x