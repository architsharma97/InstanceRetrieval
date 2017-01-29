import os
import random

random.seed(23)

folder_list = os.listdir('../Dataset')
data_dict = {}

for each in folder_list:
    data_dict[each] = os.listdir('../Dataset/'+each)

train_list = []
val_list = []
for key in data_dict.keys():
    random.shuffle(data_dict[key])
    train_list = data_dict[key][:57]
    val_list = data_dict[key][57:]
    with open('../train_list.txt','a') as f:
        for a in train_list:
            f.write(key+'/'+a+'\n')
    with open('../val_list.txt','a') as f:
        for a in val_list:
            f.write(key+'/'+a+'\n')