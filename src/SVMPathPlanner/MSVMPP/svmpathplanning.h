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


#ifndef SVMPATHPLANNING_H
#define SVMPATHPLANNING_H

#include "GPULibSVM/svm.h"
// #include "nodexyzrgb.h"

#include <string.h>
#include <fstream>

// #include <pcl/point_cloud.h>
#include <pcl/common/common.h>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/visualization/cloud_viewer.h>

#include <opencv2/opencv.hpp>

#include <vector_types.h>

#include <lemon/list_graph.h>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>

#include <costmap_2d/costmap_2d_ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <nav_msgs/Path.h>
#include <tf/transform_datatypes.h>
#include <nav_core/base_global_planner.h>
#include <nav_msgs/GetPlan.h>
#include <pcl_ros/publisher.h>

#define NDIMS 2

#define FBO_SIZE 1024                    // Texture size to be used (256, 512, 1024, 2048, 4096)

using namespace std;

namespace svmpp {

extern "C"
void launchSVMPrediction(const svm_model * &model, 
                         const unsigned int & rows, const unsigned int & cols, 
                         unsigned char * &h_data);

extern "C"
void launchCheckEdges(const float2 * &pointsInMap, const unsigned int &nPointsInMap,
                      const float2 * &edgeU, const float2 * &edgeV, const unsigned int &nEdges,
                      const float &minDist, bool * &validEdges);

extern "C"
void GPUPredictWrapper(int m, int n, int k, float kernelwidth, const float *Test, 
                       const float *Svs, float * alphas,float *prediction, float beta,
                       float isregression, float * elapsed);

// const int EDGE_TYPE_SIMPLE = 0;
// const int EDGE_TYPE_DT = 1;

enum EdgeType { EDGE_TYPE_SIMPLE = 0, EDGE_TYPE_DT = 1 };

typedef double2 CornerLimitsType;
typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> PointCloudType;
typedef pcl::PointXYZRGB PointTypeExt;
typedef pcl::PointCloud<PointTypeExt> PointCloudTypeExt;

typedef lemon::ListGraph Graph;
typedef lemon::ListGraph::Node Node;
typedef lemon::ListGraph::Edge Edge;
typedef lemon::ListGraph::EdgeMap<double> EdgeMap;
typedef lemon::ListGraph::NodeMap<PointType> NodeMap;
typedef lemon::ListGraph::EdgeMap<EdgeType> EdgeTypeMap;

class SVMPathPlanning {
    
public:
    SVMPathPlanning();
    SVMPathPlanning ( const SVMPathPlanning& other );
    virtual ~SVMPathPlanning();
    
    void testSingleProblem();
    void obtainGraphFromMap(const PointCloudType::Ptr & inputCloud, const bool & visualize);
    
    bool findShortestPath(const PointType& start, const double & startOrientation,
                          const PointType & goal, const double & goalOrientation,
                          PointCloudType::Ptr rtObstacles, bool visualize);
    
    void filterPath(PointCloudType::Ptr & path);
    
    void getGraph(PointCloudType::Ptr & pointCloud);
    
    bool isMapGenerated() { return m_mapGenerated; }
    
    PointCloudType::Ptr getPath() { return m_path; }
    
    void setSVMParams(const struct svm_parameter & params) { m_param = params; }
    
    void setMinPointDistance(const double & minPointDistance) { m_minPointDistance = minPointDistance; }
    void setMinDistBetweenObstacles(const double & minDistBetweenObstacles) { m_minDistBetweenObstacles = minDistBetweenObstacles; }
    void setMinDistCarObstacle(const double & minDistCarObstacle) { m_minDistCarObstacle = minDistCarObstacle; }
    void setCarWidth(const double & carWidth) { m_carWidth = carWidth; }
    void setInflatedMap(const cv::Mat & inflatedMap, const PointType & origin, const double & resolution) { 
        inflatedMap.copyTo(m_inflatedMap); 
        m_origin = origin;
        m_resolution = resolution;
    }
    
protected:
    void addLineToPointCloud(const PointType& p1, const PointType& p2, 
                             const uint8_t & r, const uint8_t & g, const uint8_t  & b,
                             PointCloudTypeExt::Ptr &linesPointCloud, double zOffset = 0.0);
    void getBorderFromPointClouds (PointCloudType::Ptr & X, PointCloudType::Ptr & Y,
                                   const CornerLimitsType & minCorner, const CornerLimitsType & maxCorner, 
                                   const CornerLimitsType & interval, const cv::Size & gridSize, 
                                   const uint32_t & label, PointCloudType::Ptr & pathNodes, vector<Node> & nodeList);
    void predictSVM(const svm_model * &model,
                    const unsigned int & rows, const unsigned int & cols, 
                    cv::Mat & mapPrediction);
    void getContoursFromSVMPrediction(const svm_model * &model, const CornerLimitsType & interval,
                                      const CornerLimitsType & minCorner, const CornerLimitsType & maxCorner,
                                      const cv::Size & gridSize, const uint32_t & label,
                                      PointCloudType::Ptr & pathNodes, vector<Node> & nodeList);
    
    void clusterize(const PointCloudType::Ptr & pointCloud, vector< PointCloudType::Ptr > & classes,
                    CornerLimitsType & minCorner, CornerLimitsType & maxCorner);
    
    void generateRNG(const PointCloudType::Ptr & pathNodes, vector<Node> & nodeList, const PointCloudType::Ptr & currentMap,
                     const bool & doExtendedGraph = true, const bool & doSegmentChecking = true);
    
    bool getFootPrint(const PointType & position, const double & orientation, 
                      const PointCloudType::Ptr & rtObstacles, vector<PointCloudType::Ptr> & footprint,
                      const bool & isGoal = false);
    
    void filterExistingObstacles(PointCloudType::Ptr & rtObstacles);
    
//     void visualizeClasses(const vector< PointCloudType::Ptr > & classes, const PointCloudType::Ptr & pathNodes,
//                           const PointCloudType::Ptr & rtObstacles, const PointCloudType::Ptr & path);
        
    void checkSegments(const PointCloudType::Ptr & pathNodes, vector<Node> & nodeList, 
                       const PointCloudType::Ptr & currentMap, const vector< pair<uint32_t, uint32_t> > & edges,
                       const bool & doSegmentChecking, 
                       const uint32_t & totalDelaunayEdges = std::numeric_limits<uint32_t>::max());
    
    void generateSpline(const PointType & p1, const PointType & p2, 
                         const PointType & p1Prev, const PointType & p2Next, PointCloudType::Ptr & spline);
    
    struct svm_parameter m_param;
    
    double m_minPointDistance;                  // Minimal distance between samples in downsampling
    cv::Size m_mapGridSize;
    cv::Size m_mapGridSizeRT;
    double m_minDistBetweenObstacles;          // Minimal distance between obstacles in clustering
    double m_distBetweenSamples;               // Maximal distance to be considered as an edge in the graph
    
    double m_carWidth;
    double m_minDistCarObstacle;               // Minimal distance between the car and an obstacle in path finding
    
    PointCloudType::Ptr m_originalMap;
    PointCloudType::Ptr m_pathNodes;
    
    vector< PointCloudType::Ptr > m_classes;
    
    Graph m_graph;
    boost::shared_ptr<EdgeMap> m_distMap;
    boost::shared_ptr<EdgeTypeMap> m_edgeTypeMap;
    boost::shared_ptr<NodeMap> m_nodeMap;
    vector<Node> m_nodeList;
    
    PointCloudType::Ptr m_path;
    
    CornerLimitsType m_minCorner, m_maxCorner;
    
    bool m_mapGenerated;
    
    cv::Mat m_inflatedMap;
    PointType m_origin;
    double m_resolution;
    
    // Debug
    ros::Publisher m_dbgPub;
};
    
}

#endif // SVMPATHPLANNING_H
