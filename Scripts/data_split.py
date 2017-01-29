import os
import random

random.seed(23)

train_list = os.listdir('../Dataset/train')
val_list = os.listdir('../Dataset/val')
train_dict = {}
val_dict = {}

for each in train_list:
    train_dict[each] = os.listdir('../Dataset/train/'+each)

for each in val_list:
    val_dict[each] = os.listdir('../Dataset/val/'+each)


train_list = []
val_list = []

for key in train_dict.keys():
    random.shuffle(train_dict[key])
    random.shuffle(val_dict[key])

    train_list = train_dict[key]
    val_list = val_dict[key]
    with open('../train_list.txt','a') as f:
        for a in train_list:
            f.write(key+'/'+a+'\n')
    with open('../val_list.txt','a') as f:
        for a in val_list:
            f.write(key+'/'+a+'\n')