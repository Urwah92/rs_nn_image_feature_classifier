// UIMA
#include <uima/api.hpp>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

// RS
//#include <robosherlock/utils/exception.h>
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>
#include <robosherlock/utils/common.h>

#include <robosherlock/DrawingAnnotator.h>

#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <rapidjson/document.h>

#include <ros/ros.h>
#include <rs_nn_image_feature_classifier/classifier.h>

using namespace uima;

class client : public Annotator
{
  private:
  cv::Mat  color_mat_, rgb, mask;

  ros::NodeHandle nh;   
  std::vector<cv::Rect> cluster_rois_;
  cv::Rect roi_;
  struct Cluster
  {
    size_t indices_index_;
    cv::Rect roi_, roi_hires_;
    cv::Mat mask, mask_hires_;
  };
  std::vector<Cluster> cluster;
public:


  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");
    return UIMA_ERR_NONE;
  }

  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }
private:
  TyErrorId process(CAS &tcas, ResultSpecification const &res_spec)
  {
    outInfo("process start");
    rs::StopWatch clock;
    rs::SceneCas cas(tcas);
    rs::Scene scene = cas.getScene();
    std::vector<rs::ObjectHypothesis> clusters;
    
    cas.get(VIEW_COLOR_IMAGE_HD, color_mat_);
    
    // --Jan. 2021
    // This annotator will be a testbed for a new filter method.
    // If you encounter any problems with this annotator
    // because of the usage of the new .filterOverAllSubtypes method, please report back.
    scene.identifiables.filterOverAllSubtypes(clusters);
    //scene.identifiables.filter(clusters);
    cluster_rois_.resize(clusters.size());
    std::cout<<"clusters: "<<clusters.size();
    cv::Mat disp = color_mat_.clone();

    for(size_t idx = 0; idx < clusters.size(); ++idx)
    {
      rs::ImageROI image_rois = clusters[idx].rois.get();
            //======================= Get ROI from image ==========================
      std::cout << "Image Shape"<< color_mat_<<std::endl;
      //cv::Mat rgb, mask;
      cv::Rect roi;
      rs::conversion::from(image_rois.roi(), roi);
      rs::conversion::from(image_rois.mask(), mask);
      cluster_rois_[idx] = roi;
      assert(roi.width == mask.size().width);
      assert(roi.height == mask.size().height);
      color_mat_(roi).copyTo(rgb, mask);

      //cv::rectangle(disp, clusters[idx].rois, rs::common::cvScalarColors[idx % rs::common::numberOfColors]);
    
      rs_nn_image_feature_classifier::classifier srv;
      sensor_msgs::Image image_msg;
      cv_bridge::CvImage cv_image;
      cv_image.image = rgb;
      cv_image.encoding = "bgr8";
      cv_image.toImageMsg(image_msg);
      ros::ServiceClient client = nh.serviceClient<rs_nn_image_feature_classifier::classifier>("classifier");
      srv.request.rgb = image_msg;
      std::cout << "Image Shape"<< rgb<<std::endl;
      if( client.call(srv)){

        ROS_INFO("Received");
      }
      else{
        ROS_ERROR("Failed to call service");
      }
    }
    outInfo("took: " << clock.getTime() << " ms.");
    return UIMA_ERR_NONE;
  }
    void drawImageWithLock(cv::Mat &disp)
  {
    disp = color_mat_.clone();
    for(size_t i = 0; i < cluster.size(); ++i)
    {
      cv::rectangle(disp, cluster[i].roi_, rs::common::cvScalarColors[i % rs::common::numberOfColors]);
    }
  }
};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(client)
  
