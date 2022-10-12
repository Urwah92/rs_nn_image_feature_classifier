#include <uima/api.hpp>
#include<string>

#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <robosherlock/scene_cas.h>
#include <robosherlock/DrawingAnnotator.h>
#include <robosherlock/utils/output.h>
#include <robosherlock/utils/time.h>
#include <robosherlock/utils/common.h>

#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <rs_nn_image_feature_classifier/classifier.h>

using namespace uima;

class NNImgFeatureClassifier : public DrawingAnnotator
{
  private:
  ros::NodeHandle nh;   
  cv::Mat rgb, mask;

  std::vector<cv::Scalar> colors_;
  std::vector<cv::Rect> cluster_rois_;
  std::vector<std::string> class_label;

  cv::Mat color_mat_;

  public:
  NNImgFeatureClassifier() : DrawingAnnotator(__func__)
  {
  }
  

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    return UIMA_ERR_NONE;

  }

  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }
private:
  TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec)
  {
    MEASURE_TIME;
    outInfo("process start");
    rs::SceneCas cas(tcas);
    rs::Scene scene = cas.getScene();
    std::vector<rs::ObjectHypothesis> clusters;

    cas.get(VIEW_COLOR_IMAGE, color_mat_);
    rs::Query qs = rs::create<rs::Query>(tcas);
    
    // --Jan. 2021
    // This annotator will be a testbed for a new filter method.
    // If you encounter any problems with this annotator
    // because of the usage of the new .filterOverAllSubtypes method, please report back.
    scene.identifiables.filterOverAllSubtypes(clusters);
    
    cluster_rois_.resize(clusters.size());
    class_label.clear();
    for(size_t idx = 0; idx < clusters.size(); ++idx)
    {
      rs::ImageROI image_rois = clusters[idx].rois.get();

      //======================= Calculate HSV image ==========================

      cv::Rect roi;
      rs::conversion::from(image_rois.roi(), roi);
      rs::conversion::from(image_rois.mask(), mask);
      cluster_rois_[idx] = roi;
      assert(roi.width == mask.size().width);
      assert(roi.height == mask.size().height);
      color_mat_(roi).copyTo(rgb, mask);

      rs_nn_image_feature_classifier::classifier srv;
      sensor_msgs::Image image_msg;
      cv_bridge::CvImage cv_image;
      cv_image.image = rgb;
      cv_image.encoding = "bgr8";
      cv_image.toImageMsg(image_msg);
      ros::ServiceClient client = nh.serviceClient<rs_nn_image_feature_classifier::classifier>("classifier");
      srv.request.rgb = image_msg;
      //class_label[idx] = srv.response.label;
      
      if( client.call(srv)){
        ROS_INFO("Received");
        //class_label[idx] = srv.response.label;
        //if(class_label.size() <= clusters.size()){
        class_label.push_back(srv.response.label);
        //}
      }
      else{
        ROS_ERROR("Failed to call service");
      }
    }

    return UIMA_ERR_NONE;
    
  } 
  void drawImageWithLock(cv::Mat &disp)
  {
    disp = color_mat_.clone();
    for(size_t i = 0; i < cluster_rois_.size(); ++i)
    {
      const cv::Rect &roi = cluster_rois_[i];
      //Point text_position(0, 0);//Declaring the text position//
      int font_size = 1;//Declaring the font size//
      //Scalar font_Color(0, 0, 0);//Declaring the color of the font//
      int font_weight = 1;//Declaring the font weight//
      cv::putText(disp, class_label[i], cv::Point(roi.x,roi.y),cv::FONT_HERSHEY_COMPLEX_SMALL, font_size,cv::Scalar(0,255,0), font_weight,false);

      cv::rectangle(disp, roi, rs::common::cvScalarColors[i % rs::common::numberOfColors]);
      
    }
    //class_label.clear();

  }

};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(NNImgFeatureClassifier)
  
