Introduction:

It is a ros package that is compatible with robosherlock. In this package, there is a client and a server. The client node (NNImgFeatureClassifier.cpp) extracts images from the robosherlock and sends image data to a service node (classifier.py), and the service node extracts features from the image using the MobileNet keras model and uses those features in the KNN algorithm. Then it returns the class id of the image, its confidence list, and the class name. The client node then shows these class names on the robosherlock image.

Working Guide:
1. NNImgFeatureClassifier.cpp can be found in the src folder.
2. Feature_extract.py extracts features from the dataset and saves them in the feature.npy file along with the class ids. It also saves the list of class names available in the dataset with their corresponding ids in classes.csv.

     •	Include the dataset in the package and add the path in line 86 (label_dir).
      
              label_dir = package_path + "/data/partial_views"
      

3. Classifier.py is a service node which uses the features data from the features.npy file and class names from classes.csv.
Accuracy_test.py is to check the accuracy of the validation dataset.
  
     •	Add path at line 18: 
     
               label_dir = package_path + "/dataset/val_data":
