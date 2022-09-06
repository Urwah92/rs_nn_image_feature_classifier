from sklearn.neighbors import KNeighborsClassifier
from rs_nn_image_feature_classifier.srv import classifier, classifierResponse
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from cv_bridge import CvBridge, CvBridgeError
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import rospkg
import rospy
import cv2
import os

def get_image_feature(bgr_image):
  """Extract the features of rgb_image using MobileNet model"""
  bgr_image= cv2.resize(bgr_image,(224,224))
  
  print('Received Image Shape: {}'.format(bgr_image.shape))
  rgb_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)
  cv2.imshow("image",rgb_image)
  cv2.waitKey(0)
  reformed_img= Image.fromarray(rgb_image)
  array_img = np.array(reformed_img,dtype=float) 
  print(array_img.shape)                   
  reshaped_img = np.expand_dims(array_img, axis=0)
  print(reshaped_img.shape)
  reshaped_img /= 255.0
  
  #resize_img = rgb_image.reshape((1,rgb_image.shape[0],
           #                         rgb_image.shape[1],
         #
         #                            rgb_image.shape[2]))

  # prepare the image for the MobileNet model
  #image = preprocess_input(resize_img)

  model = MobileNet(weights= 'imagenet', input_shape=(224, 224,3))
  model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
  feature = model.predict(reshaped_img)
  print('Feature shape of an Image: {}'.format(feature.shape))
  return feature


def get_model_feature(file_path):
  """ Load model features from the file """
  with open(file_path, 'rb') as f:
    model_features = np.load(f, allow_pickle=True)
  print('knn_classifier features shape: {}'.format(model_features.shape))
  return model_features


def remove_labels(model_features):
  """Removes the class_ids from the loaded feature file, class_id
     exists at the end of each row  """
  formated_data = []
  labels = []
  for z in model_features:
    fix_data = z[:-1]
    clip_label = z[-1]
    labels.append(clip_label)
    formated_data.append(fix_data)
  labels_list = np.array(labels)
  formated_data = np.array(formated_data)
  print('Resized Feature Vector: {} '.format(formated_data.shape))
  return formated_data, labels_list


def reshape_label_array(no_classes, rows, label_list):
  """Label list is reshaped (dataset size, number of classes)"""
  label_length= no_classes * rows
  loaded_labels= [0] * label_length
  loaded_labels= np.array(loaded_labels)
  reshaped_label = np.reshape(loaded_labels,(-1,no_classes))

  for count,j in enumerate(label_list):
    reshaped_label[count][int(j)]= 1
  print('Reshaped Label Array: {} '.format(reshaped_label.shape))
  return reshaped_label

def val_function(val_path):
    model = MobileNet(weights= 'imagenet',input_shape = (224, 224,3))
    model.summary()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    val_datagen = ImageDataGenerator(rescale = 1./255)
    test_it = val_datagen.flow_from_directory(val_path, class_mode='categorical', batch_size= 500, target_size=(224,224))
    val_x,val_y = test_it.next()

    val_features = model.predict(val_x, batch_size=50)

    return val_features,val_y 

def knn_classifier(model_features, image_feature, label_arr):
  """Use KNN classifier to predict
    returns: predicted class_id and confidence of each class"""
  classifier_model = KNeighborsClassifier(n_neighbors=3)
  classifier_model.fit(model_features,label_arr)
  y_pred = classifier_model.predict(image_feature)
  return y_pred

#Dataset path
rospack = rospkg.RosPack()
package_path = rospack.get_path('rs_nn_image_feature_classifier')
label_dir = package_path + "/dataset/val_data"
#label_base_dir = os.listdir(label_dir)

#Saved file path
path= package_path + "/scripts/features.npy"

feature_data = get_model_feature(path)

formated_data, classes = remove_labels(feature_data)

no_classes = 20
rows, _ = feature_data.shape
reshaped_label = reshape_label_array(no_classes, rows, classes)

val_feature, val_y = val_function(label_dir)

class_id = knn_classifier(formated_data, val_feature, reshaped_label)
print(formated_data.shape)
print(classes.shape)
print(val_feature.shape)
print(class_id.shape)
print(val_y.shape)
scores = metrics.accuracy_score(val_y, class_id)
print(scores)
print(metrics.confusion_matrix(val_y.argmax(axis=1),class_id.argmax(axis=1)))