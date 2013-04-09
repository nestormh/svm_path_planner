#include <iostream>

#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/cloud_viewer.h>

#include<opencv2/opencv.hpp>

#include <boost/graph/graph_traits.hpp>
  #include <boost/graph/adjacency_list.hpp>

#include "svmpathplanning.h"

#define MAP_BASE "/home/nestor/Dropbox/projects/MultiSVMPathPlanning/maps/parkingETSII1.pgm"
#define REAL_TIME "/home/nestor/Dropbox/projects/MultiSVMPathPlanning/maps/realTime/realTime1.pgm"

// Multiclass SVM in CUDA
// http://code.google.com/p/multisvm/source/checkout
// http://patternsonascreen.net/cuSVM.html

typedef svmpp::PointType SVMPointType;
typedef svmpp::PointCloudType SVMPointCloud;

void map2PointCloud(const cv::Mat & map, double resolution, 
                    SVMPointCloud::Ptr & pointCloud) {

    pointCloud = SVMPointCloud::Ptr(new SVMPointCloud);
    pointCloud->reserve(map.rows * map.cols);

    for (uint32_t i = 0, idx = 0; i < map.rows; i++) {
        for (uint32_t j = 0; j < map.rows; j++, idx++) {
            SVMPointType point;
            if (map.data[idx] == 0) {
                point.x = j * resolution;
                point.y = i * resolution;
                
                pointCloud->push_back(point);
            }
        }
    }
}

void getFootPrint(const svmpp::PointType & position, const double & width,
                  svmpp::PointCloudType::Ptr & footprint) {
    
    footprint = svmpp::PointCloudType::Ptr(new svmpp::PointCloudType);
    footprint->reserve(360);
    
    for (double alpha = 0.0; alpha < 2 * M_PI; alpha += M_PI / 180) {
        svmpp::PointType point(width * cos(alpha) + position.x, 
                               width * sin(alpha) + position.y,
                               0.0);
        footprint->push_back(point);
    }
}

int main(int argc, char **argv) {
    double resolution = 0.1;
    double carWidth = 1.0;
    
    cv::Mat map = cv::imread(MAP_BASE, 0);
    cv::Mat rtObstaclesMap = cv::imread(REAL_TIME, 0);
    
    SVMPointCloud::Ptr pointCloud;
    SVMPointCloud::Ptr rtObstacles;
    svmpp::PointCloudType::Ptr footprint;
    
    map2PointCloud(map, resolution, pointCloud);
    map2PointCloud(rtObstaclesMap, resolution, rtObstacles);
    
    svmpp::PointType start, goal;
    goal.x = 50.0;
    goal.y = 50.0;
    
    start.x = 886 * resolution;
    start.y = 748 * resolution;
    
    getFootPrint(start, carWidth, footprint);
    
//     cv::resize(map, map, cv::Size(800, 800));
//     cv::imshow("Map", map);
//     cv::waitKey(200);
    
    svmpp::SVMPathPlanning pathPlanner;
    pathPlanner.obtainGraphFromMap(pointCloud);
//     start.x = 1; start.y = 2;
//     goal.x = 4; goal.y = 2;
    
    pathPlanner.findShortestPath(start, goal, footprint, rtObstacles);
//     pathPlanner.testDijkstra();
    pathPlanner.visualizeClasses();
    
    return 0;
}
