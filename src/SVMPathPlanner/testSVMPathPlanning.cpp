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

#include <pcl/io/pcd_io.h>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/cloud_viewer.h>

#include<opencv2/opencv.hpp>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>

#include "MSVMPP/svmpathplanning.h"
#include "MSVMPP/svmpathplanningsingle.h"
#include "MSVMPP/voronoipathplanning.h"
#include "MSVMPP/voronoisvmpathplanning.h"

// #define MAP_BASE "/home/nestor/Dropbox/projects/MultiSVMPathPlanning/maps/mapaMiura.pgm"
#define MAP_BASE "/home/nestor/Dropbox/projects/MultiSVMPathPlanning/maps/parkingETSII1.pgm"
// #define MAP_BASE "/home/nestor/Dropbox/projects/MultiSVMPathPlanning/maps/parkingETSII1WithoutPath.pgm"
// #define MAP_BASE "/home/nestor/Dropbox/projects/MultiSVMPathPlanning/maps/mapaSeparado.pgm"
#define REAL_TIME "/home/nestor/Dropbox/projects/MultiSVMPathPlanning/maps/realTime/realTime1.pgm"

#define REAL_TIME_PCD "/home/nestor/Dropbox/ros/groovy/svm_path_planner/maps/rtObstacles.pcd"

void map2PointCloud(const cv::Mat & map, double resolution, 
                    svmpp::PointCloudType::Ptr & pointCloud) {

    pointCloud = svmpp::PointCloudType::Ptr(new svmpp::PointCloudType);
    pointCloud->reserve(map.rows * map.cols);

    for (uint32_t i = 0, idx = 0; i < map.rows; i++) {
        for (uint32_t j = 0; j < map.rows; j++, idx++) {
            svmpp::PointType point;
            if (map.data[idx] == 0) {
                point.x = j * resolution;
                point.y = i * resolution;
                
                pointCloud->push_back(point);
            }
        }
    }
}

int main(int argc, char **argv) {
    double resolution = 0.1;
    double carWidth = 1.0;
    
    svmpp::PointCloudType::Ptr rtObstacles(new svmpp::PointCloudType);
    const std::string rtPCDName = REAL_TIME_PCD;
    pcl::io::loadPCDFile<svmpp::PointType> (rtPCDName, *rtObstacles);
    svmpp::PointType start(1.77012, -19.5852, 0), goal(-55.699,44.6043,0);
    double startOrientation = -3.07571;
    double goalOrientation = 1.50029;
    
    /*cv::Mat map = cv::imread(MAP_BASE, 0);
    cv::Mat rtObstaclesMap = cv::imread(REAL_TIME, 0);
    
    svmpp::PointCloudType::Ptr pointCloud;
    svmpp::PointCloudType::Ptr rtObstacles;
    
    map2PointCloud(map, resolution, pointCloud);
    map2PointCloud(rtObstaclesMap, resolution, rtObstacles);
    
    svmpp::PointType start, goal, goalWithoutPath;
    start.x = 399 * resolution;
    start.y = 576 * resolution;
    
    goal.x = 886 * resolution;
    goal.y = 748 * resolution;
    
//     cv::resize(map, map, cv::Size(800, 800));
//     cv::imshow("Map", map);
//     cv::waitKey(200);
    goalWithoutPath.x = 1192 * resolution;
    goalWithoutPath.y = 374 * resolution;*/
    
//     svmpp::SVMPathPlanning pathPlanner;
//     pathPlanner.obtainGraphFromMap(pointCloud, false);
//     pathPlanner.findShortestPath(start, 0, goalWithoutPath, M_PI / 4, rtObstacles, true);
    
//     start.x = 269 * resolution;
//     start.y = 455 * resolution;
//     
//     goal.x = 327 * resolution;
//     goal.y = 39 * resolution;

        svmpp::SVMPathPlanning pathPlanner;
        pathPlanner.obtainGraphFromMap(rtObstacles, false);
        pathPlanner.findShortestPath(start, startOrientation, goal, goalOrientation, rtObstacles, true);

//     svmpp::SVMPathPlanningSingle pathPlannerSingle;
//     pathPlannerMiura.setMap(pointCloud);
//     pathPlannerSingle.findShortestPath(start, startOrientation, goal, goalOrientation, rtObstacles, true);

//     svmpp::VoronoiPathPlanning pathPlannerVoronoi;
//     pathPlannerVoronoi.findShortestPathVoronoi(start, startOrientation, goal, goalOrientation, rtObstacles, true);
 
//     svmpp::VoronoiSVMPathPlanning pathPlannerVoronoiSVM;
//     pathPlannerVoronoiSVM.findShortestPath(start, 0.0, goal, 3 * M_PI / 2, pointCloud, true);
    
    
    return 0;
}
