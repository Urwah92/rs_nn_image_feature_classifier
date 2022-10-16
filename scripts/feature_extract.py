"""Example of using the MobileNet model as a feature extraction"""
from sympy import re
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from scipy.ndimage import zoom
import rospkg
import cv2
from scipy.ndimage import gaussian_filter,rotate
import csv
import os


def set_dataset_dir(label_base_dir, label_dir):
    """This function is used to delete all text files and non-RGB files from the dataset directory."""
    labels = []
    total = []
    for i in label_base_dir:
        tmp = os.listdir(label_dir + "/" + i)
        label = [str(i)] * len(tmp)
        labels.append(label)
    for j in label_base_dir:
        images = os.listdir(label_dir + '/' + j)
        for i in images[0:]:
            path = label_dir + '/' + j + '/' + i
            total.append(i)
            if i.endswith('.txt'):
                os.remove(path)
            elif i.endswith('depthcrop.png'):
                os.remove(path)
            elif i.endswith('_20_anti_rotated.png'):
                os.remove(path)


def get_model():
    """This method is used to create a keras model and eliminates the model's final layer."""
    model = MobileNet(weights='imagenet', input_shape=(224, 224, 3))
    # remove the output layer
    model.summary()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model


def get_from_train_datagen(data):
    """This method is use to extract x,y from the datagen object."""
    lis_x = []
    lis_y = []
    for u in range(len(data)):
        lis_x.append(data[u][0])
        lis_y.append(data[u][1])
    arr_x = np.array(lis_x,dtype='object')
    arr_y = np.array(lis_y,dtype='object')
    reshaped_x = np.concatenate(arr_x)
    reshaped_y = np.concatenate(arr_y)
    return reshaped_x, reshaped_y


def train_datagenerator(label_dir):
    """This method gets training object using ImageDataGenerator()"""
    train_datagen = ImageDataGenerator(rescale=1./255)
    train = train_datagen.flow_from_directory(label_dir, 
                                              class_mode='categorical', 
                                              batch_size=500,
                                              target_size=(224, 224))
    return train


def add_labels_with_features(features,y):
    """This method appends label indices at the end of each image array."""
    tmp2 = []
    tmp_var = []
    feature_x, feature_y = features.shape
    result = np.where(y == 1)
    for count, f in enumerate(features):
        tmp_var = np.append(f, result[1][count])
        tmp2 = np.append(tmp2, tmp_var)

    final_array = np.array(tmp2)
    final_array = np.reshape(final_array, (feature_x, feature_y + 1))
    print('Reshaped feature array with labels: ', final_array.shape)
    return final_array

def write_class_dic(class_labels):
    with open('./scripts/classes.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key,value in class_labels.items():
            writer.writerow([key, value])



#Dataset path
rospack = rospkg.RosPack()
package_path = rospack.get_path('rs_nn_image_feature_classifier')
label_dir = package_path + "/dataset/partial_views_small"
label_base_dir = os.listdir(label_dir)

#Saved file path
save_file_path= package_path + "/scripts/features.npy"

#Clean dataset directory from txt, mask and other irrelevant files.
#set_dataset_dir(label_base_dir,label_dir)

model = get_model()
train = train_datagenerator(label_dir)
x, y = get_from_train_datagen(train)

print('Shape of array y:', y.shape)
print('Shape of array x:', x.shape)

features = model.predict(x, batch_size=100)
print('Predicted Features shape: ', features.shape)

set_features = add_labels_with_features(features,y)
with open(save_file_path, 'wb') as f:
  np.save(f, set_features)

write_class_dic(train.class_indices)
