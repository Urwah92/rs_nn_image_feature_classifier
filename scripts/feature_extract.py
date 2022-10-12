"""Example of using the MobileNet model as a feature extraction"""
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
###########################
def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out
    ######################

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
            #elif i.endswith('zoom.png'):
              #  os.remove(path)
            #if i.endswith('crop.png'):
                #image= cv2.imread(path)
                #image_zoom= cv2.resize(image,None,fx=1.5,fy=1.5)
                #rotated_img = rotate(image,-20)
                #zm2 = clipped_zoom(image, 1.5)
                ##kernel = np.array([[1/16, 2/16, 1/16],
                #       [2/16, 4/16,2/16],
                #      [1/16, 2/16, 1/16]])
                #image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
                #image_sharp2 = cv2.filter2D(src=image_sharp, ddepth=-1, kernel=kernel)
                #image_sharp3 = cv2.filter2D(src=image_sharp2, ddepth=-1, kernel=kernel)
                #image_sharp4 = cv2.filter2D(src=image_sharp3, ddepth=-1, kernel=kernel)
                #image= cv2.imread(path)

                #blurred_image = gaussian_filter(image,sigma=3)
                #cv2.imwrite(path[:-4]+'_zoomin.png',zm2)


def get_model():
    """This method is used to create a keras model and eliminates the model's final layer."""
    model = MobileNet(weights='imagenet', input_shape=(224, 224, 3))
    # remove the output layer
    model.summary()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model


def get_from_train_datagen(data,i):
    """This method is use to extract x,y from the datagen object."""
    z = []
    for u in range(len(data)):
        for v in range(data.batch_size):
            z.append(data[u][i][v])
    return np.array(z)


def train_datagenerator(label_dir):
    """This method gets training object using ImageDataGenerator()"""
    train_datagen = ImageDataGenerator(rescale=1./255)
    train = train_datagen.flow_from_directory(label_dir, 
                                              class_mode='categorical', 
                                              batch_size=18000,
                                              target_size=(224, 224))
    #print(train.class_indices)
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

#set_dataset_dir(label_base_dir,label_dir)

model = get_model()
train = train_datagenerator(label_dir)
x, y = train.next()
print(train.class_indices)
print(y[-1])
#x = get_from_train_datagen(train, 0)
#y = get_from_train_datagen(train,1)

print('Array y:', y.shape)
print('Array x:', x.shape)

features = model.predict(x, batch_size=100)
print('Predicted Features shape: ', features.shape)

set_features = add_labels_with_features(features,y)
with open(save_file_path, 'wb') as f:
  np.save(f, set_features)

write_class_dic(train.class_indices)