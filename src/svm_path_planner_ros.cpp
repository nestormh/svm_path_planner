/*
 *  Copyright 2013 Néstor Morales Hernández <nestor@isaatc.ull.es>
 * 
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "svm_path_planner/svm_path_planner_ros.h"
#include <pluginlib/class_list_macros.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>

#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/common/distances.h>

#include <time.h>

#include <CGAL/Line_2.h>
#include </opt/ros/indigo/include/geometry_msgs/PoseStamped.h>

//register this planner as a BaseGlobalPlanner plugin
// PLUGINLIB_DECLARE_CLASS(svm_path_planner, SVMPathPlannerROS, svmpp_ros::SVMPathPlannerROS, nav_core::BaseGlobalPlanner)
// PLUGINLIB_EXPORT_CLASS(svmpp_ros::SVMPathPlannerROS, nav_core::BaseGlobalPlanner)
PLUGINLIB_REGISTER_CLASS(SVMPathPlannerROS, svmpp_ros::SVMPathPlannerROS, nav_core::BaseGlobalPlanner);
// PLUGINLIB_REGISTER_CLASS(VerdinoLatticeGlobalPlanner, grull_ackermann_lattice_global_planner::VerdinoLatticeGlobalPlanner, nav_core::BaseGlobalPlanner);
// PLUGINLIB_DECLARE_CLASS(navfn, SVMPathPlannerROS, navfn::SVMPathPlannerROS, nav_core::BaseGlobalPlanner)

using namespace svmpp_ros;
using namespace std;

SVMPathPlannerROS::SVMPathPlannerROS() : 
    costmap_ros_(NULL), initialized_(false) {
        
}
    
SVMPathPlannerROS::SVMPathPlannerROS(std::string name, costmap_2d::Costmap2DROS* costmap_ros) 
    : costmap_ros_(NULL),  initialized_(false) {
        //initialize the planner
        initialize(name, costmap_ros);
}
    
SVMPathPlannerROS::~SVMPathPlannerROS() {
    if (doStatistics_)
        statisticsFile_.close();
}

void SVMPathPlannerROS::initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros) {
    if(! initialized_){
        cout << "Initializing..." << endl;
        
        sleep(2);
        
        costmap_ros_ = costmap_ros;
//         planner_ = boost::shared_ptr<NavFn>(new NavFn(costmap_ros->getSizeInCellsX(), costmap_ros->getSizeInCellsY()));
        
        //get an initial copy of the costmap
        costmap_ = *(costmap_ros_->getCostmap());
        
        
        ros::NodeHandle private_nh("~/" + name);

        plan_pub_ = private_nh.advertise<nav_msgs::Path>("plan", 1);
        // Publishers
        map_point_cloud_pub_ = private_nh.advertise< svmpp::PointCloudTypeExt >("mapPointCloud", 100);
        graph_point_cloud_pub_ = private_nh.advertise< svmpp::PointCloudTypeExt >("graphPointCloud", 100);
        

        //we'll get the parameters for the robot radius from the costmap we're associated with
        inscribed_radius_ = costmap_ros_->getLayeredCostmap()->getInscribedRadius();
        

        private_nh.param <std::string> ("planner_type", plannerType_, PLANNER_TYPE_MULTISVM);
        setPlanner();
        

        struct svm_parameter svm_params;
        private_nh.param <std::string> ("planner_type", plannerType_, PLANNER_TYPE_MULTISVM);
        

        private_nh.param <int> ("svm_type", svm_params.svm_type, C_SVC);        
        private_nh.param <int> ("svm_kernel_type", svm_params.kernel_type, RBF);        
        private_nh.param <int> ("svm_degree", svm_params.degree, 3);        
        private_nh.param <double> ("svm_gamma", svm_params.gamma, 150);        
        private_nh.param <double> ("svm_coef0", svm_params.coef0, 0);        
        private_nh.param <double> ("svm_nu", svm_params.nu, 0.5);        
        private_nh.param <double> ("svm_cache_size", svm_params.cache_size, 100);        
        private_nh.param <double> ("svm_C", svm_params.C, 10000);        
        private_nh.param <double> ("svm_eps", svm_params.eps, 1e-2);        
        private_nh.param <double> ("svm_p", svm_params.p, 0.1);        
        private_nh.param <int> ("svm_shrinking", svm_params.shrinking, 0);        
        private_nh.param <int> ("svm_probability", svm_params.probability, 0);  
        

        svm_params.nr_weight = 0;
        svm_params.weight_label = NULL;
        svm_params.weight = NULL;
        
        planner_->setSVMParams(svm_params);
        
        double minPointDistance, minDistBetweenObstacles, minDistCarObstacle;
        private_nh.param <double> ("minPointDistance", minPointDistance, 0.5);
        private_nh.param <double> ("minDistBetweenObstacles", minDistBetweenObstacles, 2.5);
        private_nh.param <double> ("minDistCarObstacle", minDistCarObstacle, 0.1);
        private_nh.param <double> ("threshInflation", m_threshInflation, 100);
        
        planner_->setMinPointDistance(minPointDistance);
        planner_->setMinDistBetweenObstacles(minDistBetweenObstacles);
        planner_->setMinDistCarObstacle(minDistCarObstacle);
        

        std::string statistisFileName;
        private_nh.param <bool> ("doStatistics", doStatistics_, false);
        private_nh.param <std::string> ("statisticsFile", statistisFileName, "/tmp/stats.txt");
        

        if (doStatistics_) {
            statisticsFile_.open(statistisFileName.c_str(), ios::out | ios::trunc);
            
            statisticsFile_ << "ROS_TIME\tLENGTH\tMIN_DIST_TO_OBSTACLES\tMAX_DIST_TO_OBSTACLES\tAVG_DIST_TO_OBSTACLES\t"
                            << "MIN_SMOOTH\tMAX_SMOOTH\tAVG_SMOOTH\tTIME_ELAPSED" << endl;
        }
        

        bool doStatistics_;
        std::ofstream statisticsFile_;
        

//         obtainGraphFromMap();

        //get the tf prefix
        ros::NodeHandle prefix_nh;
        tf_prefix_ = tf::getPrefixParam(prefix_nh);
        

        make_plan_srv_ =  private_nh.advertiseService("make_plan", &SVMPathPlannerROS::makePlanService, this);
        

        cout << "Initialized" << endl;
        
        initialized_ = true;

    }
    else
        ROS_WARN("This planner has already been initialized, you can't call it twice, doing nothing");
}

void SVMPathPlannerROS::getMapPointCloud(svmpp::PointCloudType::Ptr & pointCloud,
                                         const svmpp::PointType & startPoint) {
    
    pointCloud = svmpp::PointCloudType::Ptr(new svmpp::PointCloudType);
    
    cv::Mat inflatedMap = cv::Mat::zeros(cv::Size(costmap_.getSizeInCellsX(), costmap_.getSizeInCellsY()), CV_8UC1);
    
    for (uint32_t mx = 0; mx < costmap_.getSizeInCellsX(); mx ++) {
        for (uint32_t my = 0; my < costmap_.getSizeInCellsY(); my ++) {
            
            if ((costmap_.getCost(mx, my) != costmap_2d:: NO_INFORMATION) &&
                (costmap_.getCost(mx, my) != costmap_2d::FREE_SPACE)) {

                inflatedMap.at<unsigned char>(my, mx) = 255;
            }
        }
    }
    
    cv::Mat contoursMap;
    inflatedMap.copyTo(contoursMap);
    
    cv::Mat mask = cv::Mat::zeros(inflatedMap.rows + 2, inflatedMap.cols + 2, CV_8UC1); 
    cv::floodFill(inflatedMap, mask, cv::Point2d(startPoint.x, startPoint.y), 255, 
                    0, cv::Scalar(), cv::Scalar(),  4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);

    mask = mask(cv::Rect(1, 1, inflatedMap.cols, inflatedMap.rows));
    
    planner_->setInflatedMap(mask, 
                                svmpp::PointType(costmap_.getOriginX(), costmap_.getOriginY(), 0.0), 
                                costmap_.getResolution());
    
    vector<vector<cv::Point> > contours;
    cv::findContours(contoursMap, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    
    for (vector<vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); it++) {
        for (vector<cv::Point>::iterator it2 = it->begin(); it2 != it->end(); it2++) {
            if ((it2->x >= 2) && (it2->x < inflatedMap.cols - 2) &&
                (it2->y >= 2) && (it2->y < inflatedMap.rows - 2)) {
                
                double wx, wy;
                mapToWorld(it2->x, it2->y, wx, wy);
                
                const svmpp::PointType point(wx, wy, 0.0);
                
                pointCloud->push_back(point);
            }
        }
    }
}

void SVMPathPlannerROS::getLethalObstacles(svmpp::PointCloudType::Ptr & pointCloud) {
    pointCloud = svmpp::PointCloudType::Ptr(new svmpp::PointCloudType);
    
    for (uint32_t mx = 0; mx < costmap_.getSizeInCellsX(); mx ++) {
        for (uint32_t my = 0; my < costmap_.getSizeInCellsY(); my ++) {
            if (costmap_.getCost(mx, my) == costmap_2d::LETHAL_OBSTACLE) {
            
                double wx, wy;
                mapToWorld(mx, my, wx, wy);
                
                const svmpp::PointType point(wx, wy, 0.0);
                
                pointCloud->push_back(point);
            }
        }
    }
}

bool SVMPathPlannerROS::makePlanService(nav_msgs::GetPlan::Request& req, nav_msgs::GetPlan::Response& resp) {
    makePlan(req.start, req.goal, resp.plan.poses);
    
    resp.plan.header.stamp = ros::Time::now();
    resp.plan.header.frame_id = costmap_ros_->getGlobalFrameID();
    
    return true;
} 

void SVMPathPlannerROS::mapToWorld(double mx, double my, double& wx, double& wy) {
    wx = costmap_.getOriginX() + mx * costmap_.getResolution();
    wy = costmap_.getOriginY() + my * costmap_.getResolution();
}

bool SVMPathPlannerROS::makePlan(const geometry_msgs::PoseStamped& start, 
                                 const geometry_msgs::PoseStamped& goal, 
                                 std::vector<geometry_msgs::PoseStamped>& plan){
   

    // FIXME: This is just to debug the initial map generation
//     plan.clear();
//     plan.push_back(start);
//     plan.push_back(goal);
    
//     if (!planner_->isMapGenerated())
//         obtainGraphFromMap();
    
//     svmpp::PointCloudType::Ptr tmpPointCloud;
//     getMapPointCloud(tmpPointCloud);
//     publishPointCloud(costmap_ros_->getGlobalFrameID(), tmpPointCloud, map_point_cloud_pub_);
    //publish the plan for visualization purposes
    
//     return !plan.empty();
    
    boost::mutex::scoped_lock lock(mutex_);
    if(!initialized_){
        ROS_ERROR("This planner has not been initialized yet, but it is being used, please call initialize() before use");
        return false;
    }
    

    //until tf can handle transforming things that are way in the past... we'll require the goal to be in our global frame
    if(tf::resolve(tf_prefix_, goal.header.frame_id) != tf::resolve(tf_prefix_, costmap_ros_->getGlobalFrameID())){
        ROS_ERROR("The goal pose passed to this planner must be in the %s frame.  It is instead in the %s frame.", 
        tf::resolve(tf_prefix_, costmap_ros_->getGlobalFrameID()).c_str(), tf::resolve(tf_prefix_, goal.header.frame_id).c_str());
        return false;
    }
    

    if(tf::resolve(tf_prefix_, start.header.frame_id) != tf::resolve(tf_prefix_, costmap_ros_->getGlobalFrameID())){
        ROS_ERROR("The start pose passed to this planner must be in the %s frame.  It is instead in the %s frame.", 
        tf::resolve(tf_prefix_, costmap_ros_->getGlobalFrameID()).c_str(), tf::resolve(tf_prefix_, start.header.frame_id).c_str());
        return false;
    }
    

    double wx = start.pose.position.x;
    double wy = start.pose.position.y;
    

    cout << "start frame " << start.header.frame_id << endl;
    

    costmap_ = *(costmap_ros_->getCostmap());
    

    unsigned int mx, my;
    if(!costmap_.worldToMap(wx, wy, mx, my)){
        ROS_WARN("The robot's start position is off the global costmap. Planning will always fail, are you sure the robot has been properly localized?");
        return false;
    }
    
    svmpp::PointType startPoint(start.pose.position.x, start.pose.position.y, 0.0);
    svmpp::PointType goalPoint(goal.pose.position.x, goal.pose.position.y, 0.0);
    
    svmpp::PointType startPointInMap(mx, my, 0.0);

    svmpp::PointCloudType::Ptr rtPointCloud;
    if (! planner_->isMapGenerated())
        obtainGraphFromMap(startPointInMap);
    
    getMapPointCloud(rtPointCloud, startPointInMap);
    
    cout << "RT map: " << map_point_cloud_->size() << endl;
    

    double tElapsed_old;
    

    INIT_CLOCK(beginTime)
    if (! findShortestPath(startPoint, tf::getYaw(start.pose.orientation), 
                           goalPoint, tf::getYaw(goal.pose.orientation), 
                           rtPointCloud, tElapsed_old)) {
//         tElapsed = DBL_MIN;
        
        ROS_WARN("Unable to find a path.");
    }
    END_CLOCK(tElapsed, beginTime)
    
    svmpp::PointCloudType::Ptr pointCloudPath = planner_->getPath();
    cout << "Plan found: " << pointCloudPath->size() << endl;
    
    if (doStatistics_)
        doStatistics(pointCloudPath, tElapsed);
    

    plan.clear();
    plan.reserve(pointCloudPath->size());
    
    for (svmpp::PointCloudType::iterator it = pointCloudPath->begin(); it != pointCloudPath->end(); it++) {
        geometry_msgs::PoseStamped currentPose;
        currentPose.header.frame_id = costmap_ros_->getGlobalFrameID();
        currentPose.header.stamp = ros::Time::now();
    
        currentPose.pose.position.x = it->x;
        currentPose.pose.position.y = it->y;
        currentPose.pose.position.z = it->z + 2;
        
        if (it == pointCloudPath->begin()) {
            currentPose.pose.orientation.w = start.pose.orientation.w;
            currentPose.pose.orientation.x = start.pose.orientation.x;
            currentPose.pose.orientation.y = start.pose.orientation.y;
            currentPose.pose.orientation.z = start.pose.orientation.z;
        } else {
            const svmpp::PointCloudType::iterator & prevIt = it - 1;
            const tf::Quaternion & quatOrientation = tf::createQuaternionFromYaw(atan2(it->y - prevIt->y, it->x - prevIt->x));
            
            currentPose.pose.orientation.w = (double)(quatOrientation.getW());
            currentPose.pose.orientation.x = (double)(quatOrientation.getX());
            currentPose.pose.orientation.y = (double)(quatOrientation.getY());
            currentPose.pose.orientation.z = (double)(quatOrientation.getZ());
        }

        
        plan.push_back(currentPose);
    }

    //publish the plan for visualization purposes
    if (plan.size() != 0) publishPlan(plan, 0.0, 1.0, 0.0, 0.0);
    svmpp::PointCloudType::Ptr mapPointCloud(new svmpp::PointCloudType);
    *mapPointCloud += *rtPointCloud;
    publishPointCloud(costmap_ros_->getGlobalFrameID(), mapPointCloud, map_point_cloud_pub_);
    svmpp::PointCloudType::Ptr graphPointCloud;
    planner_->getGraph(graphPointCloud);
    publishPointCloud(costmap_ros_->getGlobalFrameID(), graphPointCloud, graph_point_cloud_pub_);
    

    return !plan.empty();
}

void SVMPathPlannerROS::publishPlan(const std::vector<geometry_msgs::PoseStamped>& path, double r, double g, double b, double a) {
    if(!initialized_){
        ROS_ERROR("This planner has not been initialized yet, but it is being used, please call initialize() before use");
        return;
    }
    
    //create a message for the plan 
    nav_msgs::Path gui_path;
    gui_path.poses.resize(path.size());
    
    if(!path.empty())
    {
        gui_path.header.frame_id = path[0].header.frame_id;
        gui_path.header.stamp = path[0].header.stamp;
    }
    
    // Extract the plan in world co-ordinates, we assume the path is all in the same frame
    for(unsigned int i=0; i < path.size(); i++){
        gui_path.poses[i] = path[i];
    }
    
    plan_pub_.publish(gui_path);
}

inline void SVMPathPlannerROS::publishPointCloud(const std::string & frameId, 
                                                 const svmpp::PointCloudType::Ptr & pointCloud, 
                                                 ros::Publisher & pointCloudPublish) {

//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpPointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    sensor_msgs::PointCloud2 cloudMsg;
    pcl::toROSMsg (*pointCloud, cloudMsg);
    cloudMsg.header.frame_id=frameId;
    cloudMsg.header.stamp = ros::Time();
    
//     pointCloud->header.frame_id = frameId;
//     pointCloud->header.stamp = ros::Time();
//     pointCloud->width = 1;
//     pointCloud->height = pointCloud->points.size();
//     pointCloud->is_dense = false;

    pointCloudPublish.publish(cloudMsg);

}

// TODO: Complete this with the new planners
void SVMPathPlannerROS::setPlanner() {
    cout << "plannerType = "  << plannerType_ << endl;
    if (plannerType_ == PLANNER_TYPE_MULTISVM) {
        planner_ = boost::shared_ptr<svmpp::SVMPathPlanning> (new svmpp::SVMPathPlanning);
    } else if  (plannerType_ == PLANNER_TYPE_SINGLESVM) {
        planner_ = boost::shared_ptr<svmpp::SVMPathPlanningSingle> (new svmpp::SVMPathPlanningSingle);
    } else if  (plannerType_ == PLANNER_TYPE_VORONOI) {
        planner_ = boost::shared_ptr<svmpp::VoronoiPathPlanning> (new svmpp::VoronoiPathPlanning);
    } else if  (plannerType_ == PLANNER_TYPE_VORONOISVM) {
        planner_ = boost::shared_ptr<svmpp::VoronoiSVMPathPlanning> (new svmpp::VoronoiSVMPathPlanning);
    } else {
        ROS_WARN("%s is not a valid planner type. Using %s by default", plannerType_.c_str(), PLANNER_TYPE_MULTISVM.c_str());
        planner_ = boost::shared_ptr<svmpp::SVMPathPlanning> (new svmpp::SVMPathPlanning);
    }
}

// TODO: Complete this with the new planners
void SVMPathPlannerROS::obtainGraphFromMap(const svmpp::PointType & startPoint) {

    getMapPointCloud(map_point_cloud_, startPoint);

    // Initializing the planner...
    if (plannerType_ == PLANNER_TYPE_MULTISVM) {

        cout << "Initial map: " << map_point_cloud_->size() << endl;

        publishPointCloud(costmap_ros_->getGlobalFrameID(), map_point_cloud_, map_point_cloud_pub_);

        planner_->obtainGraphFromMap(map_point_cloud_, false);

    }
}

bool SVMPathPlannerROS::findShortestPath(const svmpp::PointType& start, const double & startOrientation,
                                         const svmpp::PointType & goal, const double & goalOrientation,
                                         svmpp::PointCloudType::Ptr rtObstacles, double & tElapsed) {

//     ros::Time beginTime = ros::Time::now();
    
    if (plannerType_ == PLANNER_TYPE_MULTISVM) {
        return planner_->findShortestPath(start, startOrientation, goal, goalOrientation, rtObstacles, false);
    } else if  (plannerType_ == PLANNER_TYPE_SINGLESVM) {
        svmpp::SVMPathPlanningSingle * singlePathPlanner = (svmpp::SVMPathPlanningSingle*)(planner_.get());
        return singlePathPlanner->findShortestPath(start, startOrientation, goal, goalOrientation, rtObstacles, false);
    } else if  (plannerType_ == PLANNER_TYPE_VORONOI) {
        svmpp::VoronoiPathPlanning * voronoiPlanner = (svmpp::VoronoiPathPlanning*)(planner_.get());
        return voronoiPlanner->findShortestPath(start, startOrientation, goal, goalOrientation, rtObstacles, false);
    } else if  (plannerType_ == PLANNER_TYPE_VORONOISVM) {
        svmpp::VoronoiSVMPathPlanning * voronoiSVMPlanner = (svmpp::VoronoiSVMPathPlanning*)(planner_.get());
        return voronoiSVMPlanner->findShortestPath(start, startOrientation, goal, goalOrientation, rtObstacles, false);
    }
    
//     ros::Duration elapsedTime = ros::Time::now() - beginTime;
//     tElapsed = elapsedTime.toNSec() / 1e6;
}

void SVMPathPlannerROS::doStatistics(const svmpp::PointCloudType::Ptr & path, const double & tElapsed) {
    double length = 0.0;
    double minDistanceToObstacles = DBL_MAX;
    double maxDistanceToObstacles = DBL_MIN;
    double avgDistanceToObstacles = 0.0;
    double minSmooth = DBL_MAX;
    double maxSmooth = DBL_MIN;
    double avgSmooth = 0.0;
    
    svmpp::PointCloudType::Ptr lethalObstacles;
    getLethalObstacles(lethalObstacles);
    
    vector<int> idxNN;
    vector<float> distNN;
    
    // Creating the KdTree object
    pcl::search::KdTree<svmpp::PointType>::Ptr treeMap (new pcl::search::KdTree<svmpp::PointType>);
    treeMap->setInputCloud (lethalObstacles);
    treeMap->setSortedResults(true);
    
    for (svmpp::PointCloudType::iterator it = path->begin(); it != path->end(); it++) {
        if (it != path->begin()) {
            length += pcl::euclideanDistance(*it, *(it - 1));
            
            if (it != path->end() - 2) {
                double slope1 = 1.0, slope2 = 1.0;
                if (it->x != (it - 1)->x)
                    slope1 = (it->y - (it - 1)->y) / (it->x - (it - 1)->x);
                if ((it + 1)->x != it->x)
                    slope2= ((it + 1)->y - it->y) / ((it + 1)->x - it->x);
                
                double sqrSlope = (slope2 - slope1) * (slope2 - slope1);
                
                if (minSmooth > sqrSlope)
                    minSmooth = sqrSlope;
                
                if (maxSmooth < sqrSlope)
                    maxSmooth = sqrSlope;
                
                avgSmooth += sqrSlope;
            }
        }
        
        treeMap->nearestKSearch(*it, 1, idxNN, distNN);
        
        if (distNN[0] < minDistanceToObstacles)
            minDistanceToObstacles = distNN[0];
        
        if (distNN[0] > maxDistanceToObstacles)
            maxDistanceToObstacles = distNN[0];
        
        avgDistanceToObstacles += distNN[0];
        
    }
    
    avgDistanceToObstacles /= path->size();
    avgSmooth /= path->size() - 2;
    
    statisticsFile_ << ros::Time::now() << 
                    "\t" << length << 
                    "\t" << minDistanceToObstacles << 
                    "\t" << maxDistanceToObstacles << 
                    "\t" << avgDistanceToObstacles << 
                    "\t" << minSmooth << 
                    "\t" << maxSmooth << 
                    "\t" << avgSmooth << 
                    "\t" << tElapsed << endl;
}