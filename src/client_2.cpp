// UIMA
#include <uima/api.hpp>

// OpenCV
#include <opencv2/opencv.hpp>

// RS
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>

#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <ros/ros.h>
#include <rs_nn_image_feature_classifier/classifier.h>

using namespace uima;

class client : public Annotator
{
  private:
  cv::Mat rgb;
  ros::NodeHandle nh;
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

  TyErrorId process(CAS &tcas, ResultSpecification const &res_spec)
  {
    outInfo("process start");
    rs::StopWatch clock;
    rs::SceneCas cas(tcas);
    cas.get(VIEW_COLOR_IMAGE_HD, rgb);
    rs_nn_image_feature_classifier::classifier srv;
    sensor_msgs::Image image_msg;
    cv_bridge::CvImage cv_image;
    cv_image.image = rgb;
    cv_image.encoding = "bgr8";
    cv_image.toImageMsg(image_msg);
    ros::ServiceClient client = 		  nh.serviceClient<rs_nn_image_feature_classifier::classifier>("classifier");
    srv.request.rgb = image_msg;
    if( client.call(srv)){

    	ROS_INFO("Received");
    }
    else{
      ROS_ERROR("Failed to call service");
    }
    outInfo("took: " << clock.getTime() << " ms.");
    return UIMA_ERR_NONE;
  }
};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(client)
  
