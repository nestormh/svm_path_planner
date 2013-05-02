/*
 *  Copyright (c) 2013, Néstor Morales Hernández <nestor@isaatc.ull.es>
 *  All rights reserved.
 * 
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 * Neither the name of the <organization> nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY NÃ©stor Morales HernÃ¡ndez <nestor@isaatc.ull.es> ''AS IS'' AND ANY
 *  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL NÃ©stor Morales HernÃ¡ndez <nestor@isaatc.ull.es> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/cloud_viewer.h>

#include<opencv2/opencv.hpp>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>

#include "svmpathplanning.h"
#include "svmpathplanningmiura.h"
#include "voronoipathplanning.h"
#include "voronoisvmpathplanning.h"

// #define MAP_BASE "/home/nestor/Dropbox/projects/MultiSVMPathPlanning/maps/mapaMiura.pgm"
#define MAP_BASE "/home/nestor/Dropbox/projects/MultiSVMPathPlanning/maps/parkingETSII1.pgm"
// #define MAP_BASE "/home/nestor/Dropbox/projects/MultiSVMPathPlanning/maps/parkingETSII1WithoutPath.pgm"
// #define MAP_BASE "/home/nestor/Dropbox/projects/MultiSVMPathPlanning/maps/mapaSeparado.pgm"
#define REAL_TIME "/home/nestor/Dropbox/projects/MultiSVMPathPlanning/maps/realTime/realTime1.pgm"

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

int main(int argc, char **argv) {
    double resolution = 0.1;
    double carWidth = 1.0;
    
    cv::Mat map = cv::imread(MAP_BASE, 0);
    cv::Mat rtObstaclesMap = cv::imread(REAL_TIME, 0);
    
    SVMPointCloud::Ptr pointCloud;
    SVMPointCloud::Ptr rtObstacles;
    
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
    goalWithoutPath.y = 374 * resolution;
    
//     svmpp::SVMPathPlanning pathPlanner;
//     pathPlanner.obtainGraphFromMap(pointCloud, false);
//     pathPlanner.findShortestPath(start, 0, goalWithoutPath, M_PI / 4, rtObstacles, true);
    
//     start.x = 269 * resolution;
//     start.y = 455 * resolution;
//     
//     goal.x = 327 * resolution;
//     goal.y = 39 * resolution;

//     svmpp::SVMPathPlanningMiura pathPlannerMiura;
//     pathPlannerMiura.setMap(pointCloud);
//     pathPlannerMiura.findShortestPath(start, 0.0, goal, 3 * M_PI / 2, pointCloud, true);

//     svmpp::VoronoiPathPlanning pathPlannerVoronoi;
//     pathPlannerVoronoi.findShortestPathVoronoi(start, 0.0, goal, 3 * M_PI / 2, pointCloud, true);
 
    svmpp::VoronoiSVMPathPlanning pathPlannerVoronoiSVM;
    pathPlannerVoronoiSVM.findShortestPath(start, 0.0, goal, 3 * M_PI / 2, pointCloud, true);
    
    
    return 0;
}
