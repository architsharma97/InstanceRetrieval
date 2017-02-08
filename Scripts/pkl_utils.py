import cPickle as pickle

# general function to save python objects using pickle
def save_obj(obj, address):
	with open(address, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# general function to load python objects using pickle
def load_obj(address):
	with open(address, 'rb')as f:
		return pickle.load(f)