#!/usr/bin/env python3
#ROS server use to detect object using KNN classifier
from sklearn.neighbors import KNeighborsClassifier
from rs_nn_image_feature_classifier.srv import classifier, classifierResponse
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from cv_bridge import CvBridge, CvBridgeError
from keras.models import Model
from PIL import Image
import numpy as np
import rospkg
import rospy
import cv2


def get_image_feature(bgr_image):
  """Extract the features of rgb_image using MobileNet model"""
  print(bgr_image)
  bgr_image= cv2.resize(bgr_image,(224,224))
  
  print('Received Image Shape: {}'.format(bgr_image.shape))
  rgb_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)
  reformed_img= Image.fromarray(rgb_image)
  array_img = np.array(reformed_img,dtype=float)                    
  reshaped_img = np.expand_dims(array_img, axis=0)
  reshaped_img /= 255.0

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


def reshape_label_array(feature_data, label_list):
  """Label list is reshaped (dataset size, number of classes)"""
  no_classes = int(max(label_list) + 1)
  rows, columns = feature_data.shape
  label_length= no_classes * rows
  loaded_labels= [0] * label_length
  loaded_labels= np.array(loaded_labels)
  reshaped_label = np.reshape(loaded_labels,(-1,no_classes))

  for count,j in enumerate(label_list):
    reshaped_label[count][int(j)]= 1
  print('Reshaped Label Array: {} '.format(reshaped_label.shape))
  return reshaped_label

def knn_classifier(model_features, image_feature, label_arr):
  """Use KNN classifier to predict
    returns: predicted class_id and confidence of each class"""
  classifier_model = KNeighborsClassifier(n_neighbors=3)
  classifier_model.fit(model_features,label_arr)
  y_pred = classifier_model.predict(image_feature)
  confidence = classifier_model.predict_proba(image_feature)
  class_confidence=[]
  confidence= list(confidence)
  for x in confidence:
    class_confidence.append(x[0][1])
  print('Class_id: {} '.format(y_pred[0]))
  print('class_conffidence: {}'.format(class_confidence))
  print(np.where(y_pred==1))
  return y_pred[0], class_confidence, np.where(y_pred==1)[1]


def shutdown_fun():
  print('shutting down')


def handle_request(request):
  """This method is called when a service request is received"""
  
  # Get package path
  rospack = rospkg.RosPack()
  package_path = rospack.get_path('rs_nn_image_feature_classifier')
  
  # Path of the feature.npy file
  path = package_path + '/scripts/features.npy'
  
  # Load data from service request
  ros_rgb_image   = request.rgb

  # Convert images to openCV
  cv_rgb_image   = None
  bridge = CvBridge()

  try:
    cv_rgb_image = bridge.imgmsg_to_cv2(ros_rgb_image)
  except CvBridgeError as e:
    print(e)
  
  img_feature = get_image_feature(cv_rgb_image)

  feature_data = get_model_feature(path)

  formated_data, classes = remove_labels(feature_data)

  reshaped_label = reshape_label_array(formated_data, classes)

  class_id, confidence, label_index = knn_classifier(formated_data, img_feature, reshaped_label)


  response= classifierResponse()
  response.class_ids = list(class_id)
  response.success = True
  response.class_confidence = list(confidence)
  class_labels = {'AlbiHimbeerJuice': 0, 'BlackCeramicBowl': 1, 'BlueCampingCup': 2, 
                  'BlueCeramicIkeaMug': 3, 'BlueMetalPlateWhiteSpeckles': 4, 'BluePlasticBowl': 5, 
                  'BluePlasticFork': 6, 'BluePlasticKnife': 7, 'BluePlasticSpoon': 8, 
                  'ComdoCappuccinoClassico': 9, 'CupEcoOrange': 10, 'EdekaRedBowl': 11, 
                  'ElBrygCoffee': 12, 'FryingPan': 13, 'FryingSpatula': 14, 'HelaCurryKetchup': 15, 
                  'JaMilch': 16, 'JodSalz': 17, 'KelloggsCornFlakes': 18, 'KelloggsToppasMini': 19, 
                  'KnusperSchokoKeks': 20, 'KoellnMuesliKnusperHonigNuss': 21, 'LargeGreySpoon': 22, 
                  'LinuxCup': 23, 'LionCerealBox': 24, 'MarkenSalz': 25, 'MeerSalz': 26, 
                  'MondaminPancakeMix': 27, 'NesquikCereal': 28, 'PfannerGruneIcetea': 29, 
                  'PfannerPfirsichIcetea': 30, 'PopcornPot': 31, 'PringlesPaprika': 32, 'PringlesSalt': 33, 
                  'PringlesVinegar': 34, 'RedMetalBowlWhiteSpeckles': 35, 'RedMetalCupWhiteSpeckles': 36, 
                  'RedMetalPlateWhiteSpeckles': 37, 'RedPlasticFork': 38, 'RedPlasticKnife': 39, 
                  'RedPlasticSpoon': 40, 'ReineButterMilch': 41, 'SeverinPancakeMaker': 42, 
                  'SiggBottle': 43, 'SlottedSpatula': 44, 'SojaMilch': 45, 'SpitzenReis': 46, 
                  'Toaster': 47, 'TomatoAlGustoBasilikum': 48, 'TomatoSauceOroDiParma': 49, 
                  'VollMilch': 50, 'WeideMilchSmall': 51, 'WhiteBottle': 52, 'WhiteCeramicIkeaBowl': 53, 
                  'WhitePitcher': 54, 'YcbFork': 55, 'YcbKnife': 56, 'YellowCeramicPlate': 57, 'YellowPitcher': 58}
  if label_index.size > 0:
    key = [k for k, v in class_labels.items() if v == label_index]
    print("Key: ", str(key[0]))
    response.label= str(key[0])
  elif label_index.size == 0:
    response.label= "no_object"
  return response


def classifier_server():
  rospy.init_node('classifier')
  s= rospy.Service('classifier', classifier, handle_request)
  rospy.on_shutdown(shutdown_fun)
  print('Ready to accept image')
  rospy.spin()



if __name__ == '__main__':
  try:
    classifier_server()
  except rospy.ROSInterruptException:
    pass
