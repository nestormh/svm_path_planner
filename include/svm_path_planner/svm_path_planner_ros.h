#ifndef SVM_PATH_PLANNER_ROS_H_
#define SVM_PATH_PLANNER_ROS_H_

#include <ros/ros.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <nav_msgs/Path.h>
#include <tf/transform_datatypes.h>
#include <vector>
#include <nav_core/base_global_planner.h>
#include <nav_msgs/GetPlan.h>
#include <pcl_ros/publisher.h>

#include "MSVMPP/svmpathplanning.h"
#include "MSVMPP/voronoipathplanning.h"
#include "MSVMPP/voronoisvmpathplanning.h"
#include "MSVMPP/svmpathplanningsingle.h"

namespace svmpp_ros {

class SVMPathPlannerROS : public nav_core::BaseGlobalPlanner {
public:
    /**
     * @brief  Default constructor for the SVMPathPlannerROS object
     */
    SVMPathPlannerROS();
    
    /**
     * @brief  Constructor for the SVMPathPlannerROS object
     * @param  name The name of this planner
     * @param  costmap_ros A pointer to the ROS wrapper of the costmap to use
     */
    SVMPathPlannerROS(std::string name, costmap_2d::Costmap2DROS* costmap_ros);
        
    /**
     * @brief  Destructor for the interface
     */
    ~SVMPathPlannerROS();
    
    /**
     * @brief  Initialization function for the BaseGlobalPlanner
     * @param  name The name of this planner
     * @param  costmap_ros A pointer to the ROS wrapper of the costmap to use for planning
     */
    void initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros);
    
    /**
     * @brief Given a goal pose in the world, compute a plan
     * @param start The start pose 
     * @param goal The goal pose 
     * @param plan The plan... filled by the planner
     * @return True if a valid plan was found, false otherwise
     */
    bool makePlan(const geometry_msgs::PoseStamped& start, 
                  const geometry_msgs::PoseStamped& goal, std::vector<geometry_msgs::PoseStamped>& plan);
                          
    bool makePlanService(nav_msgs::GetPlan::Request& req, nav_msgs::GetPlan::Response& resp);
    
    
    const std::string PLANNER_TYPE_MULTISVM = "MultiSVMPathPlanner";
    const std::string PLANNER_TYPE_SINGLESVM = "SingleSVMPathPlanner";
    const std::string PLANNER_TYPE_VORONOI = "VoronoiPathPlanner";
    const std::string PLANNER_TYPE_VORONOISVM = "VoronoiSVMPathPlanner";
private:
    void publishPlan(const std::vector<geometry_msgs::PoseStamped>& path, double r, double g, double b, double a);
    void publishPointCloud(const std::string & frameId, const pcl::PointCloud<pcl::PointXYZ>::Ptr & pointCloud, ros::Publisher & pointCloudPublish);
    
    void mapToWorld(double mx, double my, double& wx, double& wy);
    
    void getMapPointCloud(pcl::PointCloud< svmpp::PointType >::Ptr& pointCloud,
                          const svmpp::PointType & startPoint);
    void getLethalObstacles(svmpp::PointCloudType::Ptr & pointCloud);
    
    void setPlanner();
    void obtainGraphFromMap(const svmpp::PointType & startPoint);
    bool findShortestPath(const svmpp::PointType& start, const double & startOrientation,
                          const svmpp::PointType & goal, const double & goalOrientation,
                          svmpp::PointCloudType::Ptr rtObstacles, double & tElapsed);
    
    void doStatistics(const svmpp::PointCloudType::Ptr & path, const double & tElapsed);
    
    costmap_2d::Costmap2DROS* costmap_ros_;
    double inscribed_radius_;
    ros::Publisher plan_pub_;
    ros::Publisher map_point_cloud_pub_;
    ros::Publisher graph_point_cloud_pub_;
    bool initialized_;
    costmap_2d::Costmap2D costmap_;
    std::string tf_prefix_;
    boost::mutex mutex_;
    ros::ServiceServer make_plan_srv_;
    svmpp::PointCloudType::Ptr map_point_cloud_;
    
    boost::shared_ptr<svmpp::SVMPathPlanning> planner_;
    
    
    // Params
    std::string plannerType_;
    bool doStatistics_;
    std::ofstream statisticsFile_;
    double m_threshInflation;
    
};
};  
#endif
