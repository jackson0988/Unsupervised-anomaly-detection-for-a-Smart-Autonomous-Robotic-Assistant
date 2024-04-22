#include <anomaly_detection/anomaly_detection.hpp>
#include <cv_bridge/cv_bridge.h>
#include <decision_msgs/Decision.h>
#include <image_transport/image_transport.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/image_encodings.h>
#include <vector>

class Detector
{
public:
    Detector(std::string path, ros::NodeHandle* privateNh, ros::NodeHandle* publicNh) :
            privateNh(privateNh), publicNh(publicNh)
    {
        // Subscrive to input video feed and publish output video feed
        std::string image_topic, output_topic;
        double      rate;
        privateNh->param<std::string>("image_topic", image_topic, "/camera/image_raw");
        privateNh->param<std::string>("output_topic", output_topic, "/anomaly_detector/anomaly");
        privateNh->param<double>("rate", rate, 10);
        r             = ros::Rate(rate);
        it_           = new image_transport::ImageTransport(*publicNh);
        image_sub_    = it_->subscribe(image_topic, 1, &Detector::imageCb, this);
        decision_pub_ = publicNh->advertise<decision_msgs::Decision>(output_topic, 1);
        model         = path + "/saved_models/traced/epoch_30.pt";
        try {
            ad = new AnomalyDetector(model);
        }
        catch (std::exception& e) {
            ROS_ERROR("[ANOMALY] %s", e.what());
        }
        img = cv::Mat::zeros(cv::Size(128, 128), CV_8UC3);
        cvMatToVectorUChar(img, img_array);
    }

    ~Detector()
    {
        delete ad;
        delete it_;
    }

    void compute()
    {
        while (ros::ok()) {
            cvMatToVectorUChar(img, img_array);
            double conf = ad->compute(img_array);

            decision.confidence = conf;
            decision.target     = conf > 50 ? "ANOMALY" : "TAKE IT EASY";
            decision_pub_.publish(decision);
            ros::spinOnce();
            r.sleep();
        }
    }

    void cvMatToVectorUChar(cv::Mat& mat, std::vector<unsigned char>& array)
    {
        if (mat.isContinuous()) {
            array.assign(mat.data, mat.data + mat.total() * mat.channels());
        } else {
            for (int i = 0; i < mat.rows; ++i) {
                array.insert(
                    array.end(), mat.ptr<unsigned char>(i), mat.ptr<unsigned char>(i) + mat.cols * mat.channels());
            }
        }
    }

    ros::NodeHandle*                 privateNh = nullptr;
    ros::NodeHandle*                 publicNh  = nullptr;
    image_transport::ImageTransport* it_       = nullptr;
    image_transport::Subscriber      image_sub_;
    ros::Publisher                   decision_pub_;

private:
    void imageCb(const sensor_msgs::ImageConstPtr& msg)
    {
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            cv::resize(cv_ptr->image, img, cv::Size(128, 128));
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
    std::string                model;
    ros::Rate                  r  = ros::Rate(1);
    AnomalyDetector*           ad = nullptr;
    cv_bridge::CvImagePtr      cv_ptr;
    cv::Mat                    img;
    std::vector<unsigned char> img_array;
    decision_msgs::Decision    decision;
    std::mutex                 mtx;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "anomaly_detection_node");
    std::string     path = ros::package::getPath("anomaly_detection");
    ros::NodeHandle privateNh("~");
    ros::NodeHandle publicNh;
    ROS_ASSERT_MSG(!path.empty(), "[ANOMALY] Cannot find node path!");
    Detector dec(path, &privateNh, &publicNh);
    dec.compute();
    ros::spin();
    return 0;
}