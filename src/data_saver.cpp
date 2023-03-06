#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

class ImagePointCloudSaverNode
{
public:
    ImagePointCloudSaverNode()
    {
        // image_sub_f = nh_.subscribe("/theia/front_camera/color/image_raw", 1, &ImagePointCloudSaverNode::fimageCallback, this);
        image_sub_f = nh_.subscribe("/camera/color/image_raw", 1, &ImagePointCloudSaverNode::fimageCallback, this);
        image_sub_l = nh_.subscribe("/theia/right_camera/color/image_raw", 1, &ImagePointCloudSaverNode::limageCallback, this);
        image_sub_r = nh_.subscribe("/theia/left_camera/color/image_raw", 1, &ImagePointCloudSaverNode::rimageCallback, this);
        pointcloud_sub_ = nh_.subscribe("/theia/os_cloud_node/points", 1, &ImagePointCloudSaverNode::pointcloudCallback, this);

        is_key_pressed_ = false;
        counter = 176;
    }

    void fimageCallback(const sensor_msgs::Image::ConstPtr& msg)
    {
        
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        image_data_f = cv_ptr->image;
    }

    void rimageCallback(const sensor_msgs::Image::ConstPtr& msg)
    {
        
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        image_data_r = cv_ptr->image;
    }

    void limageCallback(const sensor_msgs::Image::ConstPtr& msg)
    {
        
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        image_data_l = cv_ptr->image;
    }

    void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        point_cloud_data_ = *cloud;
    }

    void saveImageAndPointCloud()
    {
        if (image_data_f.empty() || point_cloud_data_.empty() || image_data_r.empty() || image_data_l.empty())
        {
            ROS_WARN("No image or point cloud data received yet.");
            return;
        }

        std::string image_filename = "data/front_" + std::to_string(counter) + ".png";
        cv::imwrite(image_filename, image_data_f);
         image_filename = "data/right_" + std::to_string(counter) + ".png";
        cv::imwrite(image_filename, image_data_r);
         image_filename = "data/left_" + std::to_string(counter) + ".png";
        cv::imwrite(image_filename, image_data_l);
        ROS_INFO("Saved image data to %s", image_filename.c_str());

        std::string pointcloud_filename = "data/pointcloud_" + std::to_string(counter) + ".ply";
        pcl::io::savePLYFileBinary(pointcloud_filename, point_cloud_data_);
        ROS_INFO("Saved point cloud data to %s", pointcloud_filename.c_str());
        ++counter;
    }

    void waitForKey()
    {
        while (ros::ok())
        {
            
            std::cout<<"Press a key (s/q): ";
            char key;
            std::cin>>key;
            if (key == 's')
            {
                is_key_pressed_ = true;
            }
            else if (key == 'q')
            {
                ros::shutdown();
                return;
            }

            ros::spinOnce();

            if (is_key_pressed_)
            {
                this->saveImageAndPointCloud();
                is_key_pressed_ = false;
            }
        }
    }

    bool isKeyPressed()
    {
        return is_key_pressed_;
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber image_sub_f;
    ros::Subscriber image_sub_l;
    ros::Subscriber image_sub_r;
    ros::Subscriber pointcloud_sub_;

    cv::Mat image_data_f;
    cv::Mat image_data_r;
    cv::Mat image_data_l;
    pcl::PointCloud<pcl::PointXYZ> point_cloud_data_;

    bool is_key_pressed_;
    int counter;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_pointcloud_saver");
    ImagePointCloudSaverNode node;
    
    node.waitForKey();

    
    

    return 0;
}
