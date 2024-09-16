import os
import shutil
import scipy.io
import numpy as np
import time
from PIL import Image

path = '/data/pwojcik/stan_cars/'

if not os.path.isdir(path):
    os.mkdir(path)

time_start = time.time()

# convert ids to class names
class_names = scipy.io.loadmat('/data/pwojcik/stan_cars/cars_annos.mat')
class_ids_to_names = dict()
for row in range(len(class_names['class_names'][0])):
    name = class_names['class_names'][0][row][0]
    name = name.replace("/", "")  # remove slash to prevent directory errors
    name = name.replace(" ", "_")  # replace space
    class_ids_to_names[row + 1] = name

# adapted from https://github.com/tonylaioffer/cnn_car_classification/blob/master/data_prepare.py
mat = scipy.io.loadmat('/data/pwojcik/stan_cars/cars_annos.mat')
# print("annotations: ", mat['annotations'])
#training_class = mat['annotations']['class']
#training_fname = mat['annotations']['fname']
#training_x1 = mat['annotations']['bbox_x1']
#training_y1 = mat['annotations']['bbox_y1']
#training_x2 = mat['annotations']['bbox_x2']
#training_y2 = mat['annotations']['bbox_y2']

mat = scipy.io.loadmat('/data/pwojcik/stan_cars/cars_annos.mat')
# print(mat['annotations'])
#testing_class = mat['annotations']['class']
#testing_fname = mat['annotations']['fname']

training_source = '/data/pwojcik/stan_cars/cars_train/cars_train/'  # specify source training image path
training_output = path + 'train/'  # specify target trainig image path (trainig images need to be orgnized to specific structure)

testing_source = '/data/pwojcik/stan_cars/cars_test/cars_test/'  # specify source testing image path
testing_output = path + 'test/'  # specify target testing image path (testing images need to be orgnized to specific structure)


for directory in [training_output, testing_output]:
    if not os.path.exists(directory):
        os.mkdir(directory)

for idx, item in enumerate(mat['annotations'][0]):
    print(item)
    #cls = cls[0][0]
    fname = item[0][idx]
    cls = int(item[1][idx][0])
    print('!!!')
    print(fname, cls)
    fname = os.path.basename(fname)

    #train_path = os.path.join(training_source, fname)
    #test_path = os.path.join(testing_source, fname)
    #print(item, class_ids_to_names[cls], fname)

    #if os.path.exists(train_path):
    #    output_path = os.path.join(training_output, class_ids_to_names[cls])
    #else:
    #    output_path = os.path.join(testing_output, class_ids_to_names[cls])
    #if not os.path.exists(output_path):
    #    os.mkdir(output_path)
    #shutil.copy(os.path.join(training_source, fname), os.path.join(output_path, fname))


time_end = time.time()
print('Cars dataset processed, %s!' % (time_end - time_start))