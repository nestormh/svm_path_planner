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

#include "svmpathplanning.h"

#include <time.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <vector_functions.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
// #include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
// #include <pcl/visualization/cloud_viewer.h>
#include <pcl/surface/gp3.h>

#include <pcl/common/distances.h>
#include <pcl/common/transforms.h>

#include <boost/property_map/property_map.hpp>

#include <lemon/dijkstra.h>
#include <lemon/connectivity.h>

#include "gpudt/gpudt.h"

using namespace svmpp;

SVMPathPlanning::SVMPathPlanning()
{
    // default values
    m_param.svm_type = C_SVC;
    m_param.kernel_type = RBF;
    m_param.degree = 3;
    m_param.gamma = 150;//300;        // 1/num_features
    m_param.coef0 = 0;
    m_param.nu = 0.5;
    m_param.cache_size = 100;
    m_param.C = 10000;//500;
    m_param.eps = 1e-2;
    m_param.p = 0.1;
    m_param.shrinking = 0;
    m_param.probability = 0;
    m_param.nr_weight = 0;
    m_param.weight_label = NULL;
    m_param.weight = NULL;
//     cross_validation = 0;
    
    m_minPointDistance = 0.5; //2.0;
    m_mapGridSize = cv::Size(300, 300);
    m_mapGridSizeRT = cv::Size(300, 300);
    m_minDistBetweenObstacles = 2.5;

    m_pathNodes = PointCloudType::Ptr(new PointCloudType);
    m_path = PointCloudType::Ptr(new PointCloudType);
    
    m_distMap = boost::shared_ptr<EdgeMap>(new EdgeMap(m_graph));
    m_nodeMap = boost::shared_ptr<NodeMap>(new NodeMap(m_graph));
    
    m_carWidth = 1.0;
    m_minDistCarObstacle = 0.1;
    
    m_mapGenerated = false;
}

SVMPathPlanning::SVMPathPlanning ( const SVMPathPlanning& other )
{

}

SVMPathPlanning::~SVMPathPlanning()
{

}

void SVMPathPlanning::clusterize(const PointCloudType::Ptr & pointCloud, vector< PointCloudType::Ptr > & classes,
                                 CornerLimitsType & minCorner, CornerLimitsType & maxCorner) {
            
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);
    
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
    tree->setInputCloud (pointCloud);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointType> ec;
    ec.setClusterTolerance (m_minDistBetweenObstacles);
    ec.setMinClusterSize (1);
    ec.setMaxClusterSize (INT_MAX);
    ec.setSearchMethod (tree);
    ec.setInputCloud (pointCloud);
    ec.extract (cluster_indices);
    
    pcl::search::KdTree<PointType>::Ptr tree2 (new pcl::search::KdTree<PointType>);
    
    classes.reserve(classes.size() + cluster_indices.size());
    
    minCorner = make_double2(DBL_MAX, DBL_MAX);
    maxCorner = make_double2(DBL_MIN, DBL_MIN);
    
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); it++) {
        
        PointCloudType::Ptr newClass(new PointCloudType);
        newClass->reserve(it->indices.size());
        
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end (); pit++) {
            PointType point;
            point.x = pointCloud->points[*pit].x;
            point.y = pointCloud->points[*pit].y;
            
            uint32_t nPointsFound = 0;
            
            if (pit != it->indices.begin()) {
                tree2->setInputCloud (newClass);
                nPointsFound = tree2->nearestKSearch (point, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
            }
            
            if ((nPointsFound == 0) || (pointNKNSquaredDistance[0] > m_minPointDistance)) {
                if (point.x < minCorner.x) minCorner.x = point.x;
                if (point.x > maxCorner.x) maxCorner.x = point.x;
                if (point.y < minCorner.y) minCorner.y = point.y;
                if (point.y > maxCorner.y) maxCorner.y = point.y;
                
                newClass->push_back(point);
            }
        }
        
        classes.push_back(newClass);
    }
    
    maxCorner.x = 1.1 * maxCorner.x;
    maxCorner.y = 1.1 * maxCorner.y;
    minCorner.x = 0.9 * minCorner.x;
    minCorner.y = 0.9 * minCorner.y;
}

void SVMPathPlanning::addLineToPointCloud(const PointType& p1, const PointType& p2, 
                                          const uint8_t & r, const uint8_t & g, const uint8_t  & b,
                                          PointCloudTypeExt::Ptr & linesPointCloud, double zOffset) {
    
    double dist = pcl::euclideanDistance(p1, p2);
    
    const uint32_t nSamples = (uint32_t)(ceil(dist / 0.02));
    
    for (uint32_t i = 0; i <= nSamples; i++) {
        pcl::PointXYZRGB p;
        p.x = p1.x + ((double)i / nSamples) * (p2.x - p1.x);
        p.y = p1.y + ((double)i / nSamples) * (p2.y - p1.y);
        p.z = zOffset;
        
        p.r = r;
        p.g = g;
        p.b = b;
        
        linesPointCloud->push_back(p);
    } 
}

// void SVMPathPlanning::visualizeClasses(const vector< PointCloudType::Ptr > & classes, const PointCloudType::Ptr & pathNodes,
//                                        const PointCloudType::Ptr & rtObstacles = PointCloudType::Ptr(),
//                                        const PointCloudType::Ptr & path = PointCloudType::Ptr()) {
//     
//     PointCloudTypeExt::Ptr pointCloud(new PointCloudTypeExt);
//     
//     for (uint32_t i = 0; i < classes.size(); i++) {
//         PointCloudType::Ptr currentClass = classes[i];
//         
//         uchar color[] = { rand() & 255, rand() & 255, rand() & 255 };
//         
//         for (uint32_t j = 0; j < currentClass->size(); j++) {
//             
//             PointTypeExt point;
//             
//             point.x = currentClass->at(j).x;
//             point.y = currentClass->at(j).y;
//             point.z = 0.0;
//             point.r = color[0];
//             point.g = color[1];
//             point.b = color[2];
//             
//             pointCloud->push_back(point);
//         }
//     }
//     
//     PointCloudTypeExt::Ptr trajectory(new PointCloudTypeExt);
//     trajectory->reserve(pathNodes->size());
//     
//     for (uint32_t i = 0; i < pathNodes->size(); i++) {
//         PointTypeExt point;
//         
//         point.x = pathNodes->at(i).x;
//         point.y = pathNodes->at(i).y;
//         point.z = 0.0; //-1.0;
//         point.r = 0;
//         point.g = 255;
//         point.b = 0;
//         
//         trajectory->push_back(point);
//     }
// 
//     PointCloudTypeExt::Ptr pathPointCloud(new PointCloudTypeExt);
//     if (path.get() != NULL) {
//         PointType lastPoint;
//         for (uint32_t i = 0; i < path->size(); i++) {
//             const PointType & point = path->at(i);
//             
//             if (i != 0)
//                 addLineToPointCloud(lastPoint, point, 0, 255, 255, pathPointCloud, 0.0);
//             lastPoint = point;
//         }
//     }
//     
//     boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//     viewer->setBackgroundColor (0, 0, 0);
//     viewer->initCameraParameters();
//     viewer->addCoordinateSystem();
//     
//     pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgb(pointCloud);
//     viewer->addPointCloud<PointTypeExt> (pointCloud, rgb, "pointCloud");
//     
//     pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbTrajectory(trajectory);
//     viewer->addPointCloud<PointTypeExt> (trajectory, rgbTrajectory, "trajectory");
//     
// //     pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbFootprintStart(footprintStart);
// //     viewer->addPointCloud<PointTypeExt> (footprintStart, rgbFootprintStart, "footprintStart");
// //     
// //     pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbFootprintGoal(footprintGoal);
// //     viewer->addPointCloud<PointTypeExt> (footprintGoal, rgbFootprintGoal, "footprintGoal");
//     
//     if (path.get() != NULL) {
//         pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbPath(pathPointCloud);
//         viewer->addPointCloud<PointTypeExt> (pathPointCloud, rgbPath, "pathPointCloud");
//     }
//     
// //     PointCloudTypeExt::Ptr graphPointCloud(new PointCloudTypeExt);  
// //     for (Graph::ArcIt it(m_graph); it != lemon::INVALID; ++it) {
// //         Edge currentEdge;
// //         addLineToPointCloud((*m_nodeMap)[m_graph.u(it)], (*m_nodeMap)[m_graph.v(it)], 255, 0, 0, graphPointCloud);
// //     }
// //         
// //     pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbTriangulation(graphPointCloud);
// //     viewer->addPointCloud<PointTypeExt> (graphPointCloud, rgbTriangulation, "graphPointCloud");
//         
//     while (! viewer->wasStopped ()) {    
//         viewer->spinOnce();       
//     }
// }

void SVMPathPlanning::getBorderFromPointClouds (PointCloudType::Ptr & X, PointCloudType::Ptr & Y,
                                                const CornerLimitsType & minCorner, const CornerLimitsType & maxCorner, 
                                                const CornerLimitsType & interval, const cv::Size & gridSize, 
                                                const uint32_t & label, PointCloudType::Ptr & pathNodes, vector<Node> & nodeList) {
    
    struct svm_problem svmProblem;
    svmProblem.l = X->size() + Y->size();
    svmProblem.y = new double[svmProblem.l];
    svmProblem.x = new svm_node[svmProblem.l];
    
#ifdef DEBUG
    cout << "L = " << svmProblem.l << endl;
#endif
    
    for ( uint32_t i = 0; i < X->size(); i++ ) {
        const PointType & p = X->at ( i );
        
        double x = ( p.x - minCorner.x ) / interval.x;
        double y = ( p.y - minCorner.y ) / interval.y;
        
        if ((x > minCorner.x) || (y > minCorner.y) ||
            (x < maxCorner.x) || (y < maxCorner.y)) {
        
            svmProblem.y[i] = 1;
            svmProblem.x[i].dim = NDIMS;
            svmProblem.x[i].values = new double[NDIMS];
            svmProblem.x[i].values[0] = x;
            svmProblem.x[i].values[1] = y;
        }
    }

    for ( uint32_t i = 0; i < Y->size(); i++ ) {
        const PointType & p = Y->at ( i );
        double x = (p.x - minCorner.x ) / interval.x;
        double y = ( p.y - minCorner.y ) / interval.y;
        
        if ((x > minCorner.x) || (y > minCorner.y) ||
            (x < maxCorner.x) || (y < maxCorner.y)) {
            
            const int & idx = i + X->size();
            svmProblem.y[idx] = 2;
            svmProblem.x[idx].dim = NDIMS;
            svmProblem.x[idx].values = new double[NDIMS];
            svmProblem.x[idx].values[0] = x;
            svmProblem.x[idx].values[1] = y;
        }
    }
   
#ifdef DEBUG
    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    
    svm_model * model = svm_train(&svmProblem, &m_param);

#ifdef DEBUG
    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    std::cout << "Elapsed time = " << elapsed << endl;
#endif    

    getContoursFromSVMPrediction((const svm_model*&)model, interval, minCorner, maxCorner, gridSize, label, pathNodes, nodeList);
    
    svm_free_and_destroy_model(&model);
    for (uint32_t i = 0; i < svmProblem.l; i++) {
        delete svmProblem.x[i].values;
    }
    delete [] svmProblem.x;
    delete [] svmProblem.y;
}

inline void SVMPathPlanning::getContoursFromSVMPrediction(const svm_model * &model, const CornerLimitsType & interval,
                                                          const CornerLimitsType & minCorner, const CornerLimitsType & maxCorner,
                                                          const cv::Size & gridSize, const uint32_t & label,
                                                          PointCloudType::Ptr & pathNodes, vector<Node> & nodeList) {
    
    cv::Mat predictMap(gridSize, CV_8UC1);

#ifdef DEBUG
    struct timespec start, finish;
    double elapsed;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    
    launchSVMPrediction(model, gridSize.height, gridSize.width, predictMap.data);
    
#ifdef DEBUG
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    std::cout << "Elapsed time for prediction = " << elapsed << endl;
#endif
    
    vector<vector<cv::Point> > contours;
    cv::findContours(predictMap, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
            
    for (vector<vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); it++) {
        for (vector<cv::Point>::iterator it2 = it->begin(); it2 != it->end(); it2++) {
            if ((it2->x > 1) && (it2->y > 1) && (it2->x < gridSize.width - 2) && (it2->y < gridSize.height - 2) ) {
                PointType point;
                point.x = (it2->x * interval.x / gridSize.width) + minCorner.x;
                point.y = (it2->y * interval.y / gridSize.height) + minCorner.y;
                
                pathNodes->push_back(point);
            
                Node node = m_graph.addNode();
                (*m_nodeMap)[node] = point;
                nodeList.push_back(node);
            }
        }
    }
}

void SVMPathPlanning::obtainGraphFromMap(const PointCloudType::Ptr & inputCloud, const bool & visualize = false)
{
    struct timespec start, finish;
    double elapsed;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
        
    cout << __FUNCTION__ << ":" << __LINE__ << endl;
    CornerLimitsType minCorner, maxCorner, interval;
    clusterize(inputCloud, m_classes, minCorner, maxCorner);
    
    cout << __FUNCTION__ << ":" << __LINE__ << endl;
    vector< PointCloudType::Ptr >::iterator it1, it2;
    
    cout << __FUNCTION__ << ":" << __LINE__ << endl;
    interval = make_double2(maxCorner.x - minCorner.x, 
                            maxCorner.y - minCorner.y);
    m_minCorner = minCorner;
    m_maxCorner = maxCorner;
    
    m_distBetweenSamples = 1.5 * sqrt((interval.x / m_mapGridSize.width) * (interval.x / m_mapGridSize.width) +
                                      (interval.y / m_mapGridSize.height) * (interval.y / m_mapGridSize.height));
    
    cout << __FUNCTION__ << ":" << __LINE__ << endl;
    m_originalMap = PointCloudType::Ptr(new PointCloudType);
    
    cout << __FUNCTION__ << ":" << __LINE__ << endl;
    uint32_t label = 0;
    for (it1 = m_classes.begin(); it1 != m_classes.end(); it1++, label++) {
        PointCloudType::Ptr pointCloud1(new PointCloudType);
        *pointCloud1 = *(*it1);
        
        *m_originalMap += *pointCloud1;
        
        PointCloudType::Ptr pointCloud2(new PointCloudType);
        for (it2 = m_classes.begin(); it2 != m_classes.end(); it2++) {
            if (it1 != it2) {
                *pointCloud2 += *(*it2);
            }
        }
                                         
        getBorderFromPointClouds (pointCloud1, pointCloud2, minCorner, maxCorner, interval, m_mapGridSize, label, m_pathNodes, m_nodeList);
//         break;
    }
    cout << __FUNCTION__ << ":" << __LINE__ << endl;
    generateRNG(m_pathNodes, m_nodeList, m_originalMap);
    cout << __FUNCTION__ << ":" << __LINE__ << endl;
    
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    std::cout << "Total time for graph generation = " << elapsed << endl;
    
    m_mapGenerated = true;
    cout << __FUNCTION__ << ":" << __LINE__ << endl;

//     if (visualize) {
//         visualizeClasses(m_classes, m_pathNodes);
//     }
}

bool SVMPathPlanning::findShortestPath(const PointType& start, const double & startOrientation,
                                       const PointType & goal, const double & goalOrientation,
                                       PointCloudType::Ptr rtObstacles, bool visualize = false)
{
    
    struct timespec startTime, finishTime;
    double elapsed;
    
    clock_gettime(CLOCK_MONOTONIC, &startTime);
    
    CornerLimitsType minCornerRT;
    CornerLimitsType maxCornerRT;
    
    vector< PointCloudType::Ptr > classes;
    CornerLimitsType dummyCorner1, dummyCorner2;
    
    filterExistingObstacles(rtObstacles);
    cout << "After filtering: " << rtObstacles->size() << endl;
    if (rtObstacles->size() != 0)
        clusterize(rtObstacles, classes, dummyCorner1, dummyCorner2);
    
    PointCloudType::Ptr currentMap(new PointCloudType);
    *currentMap += *m_originalMap;
    *currentMap += *rtObstacles;
    
    vector<PointCloudType::Ptr> footprintStart, footprintGoal;
    bool startCheck = getFootPrint(start, startOrientation, currentMap, footprintStart);
    bool goalCheck = getFootPrint(goal, goalOrientation, currentMap, footprintGoal);
    
    if (! startCheck) {
        cerr << "Failed to find a path: Current position is too near to an obstacle or colliding with it" << endl;
        
        return false;
    }
    if (! startCheck) {
        cerr << "Failed to find a path: Goal position is not clear" << endl;
        
        return false;
    }
    filterExistingObstacles(footprintStart[0]);
    filterExistingObstacles(footprintStart[1]);
    filterExistingObstacles(footprintGoal[0]);
    filterExistingObstacles(footprintGoal[1]);
    
    if (footprintStart[0]->size() != 0) classes.push_back(footprintStart[0]);
    if (footprintStart[1]->size() != 0) classes.push_back(footprintStart[1]);
    if (footprintGoal[0]->size() != 0) classes.push_back(footprintGoal[0]);
    if (footprintGoal[1]->size() != 0) classes.push_back(footprintGoal[1]);
    
    PointCloudType::Ptr pathNodesRT(new PointCloudType);
    pcl::copyPointCloud(*m_pathNodes, *pathNodesRT);
   
    vector<Node> nodeListRT;
    copy(m_nodeList.begin(), m_nodeList.end(), back_inserter(nodeListRT));
    int maxNodeId = nodeListRT.size();
    
    if (classes.size() != 0) {
        for (Graph::EdgeIt it(m_graph); it != lemon::INVALID; ++it) {
            Edge currentEdge = it;
            m_graph.erase(currentEdge);
        }
        
        vector< PointCloudType::Ptr >::iterator it;
        uint32_t label = m_classes.size();
        for (it = classes.begin(); it != classes.end(); it++, label++) {
            CornerLimitsType minCornerRT = make_double2(DBL_MAX, DBL_MAX);
            CornerLimitsType maxCornerRT = make_double2(DBL_MIN, DBL_MIN);
            PointType centroid;
            for (PointCloudType::iterator it2 = (*it)->begin(); it2 != (*it)->end(); it2++) {
                if (it2->x < minCornerRT.x) minCornerRT.x = it2->x;
                if (it2->x > maxCornerRT.x) maxCornerRT.x = it2->x;
                if (it2->y < minCornerRT.y) minCornerRT.y = it2->y;
                if (it2->y > maxCornerRT.y) maxCornerRT.y = it2->y;
                
                centroid.x += it2->x;
                centroid.y += it2->y;
            }
            centroid.x /= (*it)->size();
            centroid.y /= (*it)->size();
            
            minCornerRT.x = min((double)centroid.x - 10, minCornerRT.x - 1.5);
            minCornerRT.y = min((double)centroid.y - 10, minCornerRT.y - 1.5);
            maxCornerRT.x = max((double)centroid.x + 10, maxCornerRT.x + 1.5);
            maxCornerRT.y = max((double)centroid.y + 10, maxCornerRT.y + 1.5);
            
            CornerLimitsType intervalRT = make_double2(maxCornerRT.x - minCornerRT.x, 
                                                    maxCornerRT.y - minCornerRT.y);
            
            getBorderFromPointClouds (*it, m_originalMap, minCornerRT, maxCornerRT, intervalRT, m_mapGridSizeRT, label, pathNodesRT, nodeListRT);
            
            //         break;
        }
        
        if (pathNodesRT->size() == 0) {
            cerr << "Unable to find a SVM decision boundary" << endl;
            return false;
        }
        
        pathNodesRT->push_back(start);
        Node nodeStart = m_graph.addNode();
        (*m_nodeMap)[nodeStart] = start;
        nodeListRT.push_back(nodeStart);
        
        pathNodesRT->push_back(goal);
        Node nodeGoal = m_graph.addNode();
        (*m_nodeMap)[nodeGoal] = goal;
        nodeListRT.push_back(nodeGoal);
        
        generateRNG(pathNodesRT, nodeListRT, currentMap);
    } else {
        nodeListRT = m_nodeList;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &finishTime);
    elapsed = (finishTime.tv_sec - startTime.tv_sec);
    elapsed += (finishTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;
    
    std::cout << "Total time for graph generation in real time = " << elapsed << endl;
        
    double distStart = DBL_MAX, distGoal = DBL_MAX;
    Node startNode, goalNode;
    PointType tmpStartPoint(start.x + m_carWidth * cos(startOrientation),
                            start.y + m_carWidth * sin(startOrientation),
                            0.0);
    PointType tmpGoalPoint(goal.x - m_carWidth * cos(goalOrientation),
                           goal.y - m_carWidth * sin(goalOrientation),
                            0.0);

    for (vector<Node>::iterator it = nodeListRT.begin(); it != nodeListRT.end(); it++) {
        
        const PointType & point = (*m_nodeMap)[*it];
        
        double tmpDistStart = pcl::squaredEuclideanDistance(tmpStartPoint, point);
        double tmpDistGoal = pcl::squaredEuclideanDistance(tmpGoalPoint, point);
        
        if (tmpDistStart < distStart) {
            distStart = tmpDistStart;
            startNode = *it;
        }
        
        if (tmpDistGoal < distGoal) {
            distGoal = tmpDistGoal;
            goalNode = *it;
        }
    }
    
    if (! lemon::connected(m_graph)) {
        Graph::NodeMap<uint32_t> compMap(m_graph);
        
        lemon::connectedComponents(m_graph, compMap);
        
        if (compMap[startNode] != compMap[goalNode]) {
            
            cerr << "Unable to find a path: It is impossible to go from " << start << 
                    " to " << goal << " in this map." << endl;
            
            return false;
        }
    }
    
    lemon::Dijkstra<Graph, EdgeMap> dijkstra(m_graph, *m_distMap);
    dijkstra.init();
    dijkstra.addSource(goalNode);
    dijkstra.start();
    
    m_path->clear();
    Node currentNode = startNode;
    m_path->push_back(start);
    while (currentNode != goalNode) {
        currentNode = dijkstra.predNode(currentNode);
        m_path->push_back((*m_nodeMap)[currentNode]);
    }
    m_path->push_back(goal);

//     if (visualize) {
//         copy(m_classes.begin(), m_classes.end(), back_inserter(classes));
//         visualizeClasses(classes, pathNodesRT, rtObstacles, m_path);
//     }
    
    return true;
}

bool SVMPathPlanning::getFootPrint(const PointType & position, const double & orientation, 
                                   const PointCloudType::Ptr & currentMap, vector<PointCloudType::Ptr> & footprint) {
    
    vector<int> idxMap, idxRT;
    vector<float> distMap, distRT;
    
    // Creating the KdTree object for the search method of the extraction
    // Create the Kd-Tree
    pcl::search::KdTree<PointType>::Ptr treeMap (new pcl::search::KdTree<PointType>);
    treeMap->setInputCloud (currentMap);
        
    treeMap->radiusSearch(position, (m_carWidth / 2 + m_minDistCarObstacle), idxMap, distMap);
    
    // If idxMap != 0, we are colliding with an obstacle
    if (idxMap.size() != 0) {
        return false;
    }
    
    footprint.resize(2);
    footprint[0] = PointCloudType::Ptr(new PointCloudType);
    footprint[0]->reserve(2 * m_carWidth / (m_minPointDistance / 2.0));
    for (double x = -m_carWidth; x <= m_carWidth; x += m_minPointDistance / 2.0) {
        footprint[0]->push_back(PointType(x, -m_carWidth, 0));
    }
    
    footprint[1] = PointCloudType::Ptr(new PointCloudType);
    footprint[1]->reserve(2 * m_carWidth / (m_minPointDistance / 2.0));
    for (double x = -m_carWidth; x <= m_carWidth; x += m_minPointDistance / 2.0) {
        footprint[1]->push_back(PointType(x, m_carWidth, 0));
    }
    
    Eigen::Affine3f tMatrix; 
    pcl::getTransformation (position.x, position.y, position.z, 0.0, 0.0, orientation, tMatrix); 
    pcl::transformPointCloud(*footprint[0], *footprint[0], tMatrix);   
    pcl::transformPointCloud(*footprint[1], *footprint[1], tMatrix);   
    
    return true;
}

void SVMPathPlanning::filterExistingObstacles(PointCloudType::Ptr & rtObstacles) {
    vector<int> pointIdxNKNSearch;
    vector<float> pointNKNSquaredDistance;
    
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
    tree->setInputCloud (m_originalMap);
    
    vector<int> inliers;
    inliers.reserve(rtObstacles->size());
    
    for (uint32_t i = 0; i < rtObstacles->size(); i++) {
        tree->radiusSearch(rtObstacles->at(i), m_minDistBetweenObstacles / 2.0, pointIdxNKNSearch, pointNKNSquaredDistance);
        
        if (pointIdxNKNSearch.size() == 0) {
            inliers.push_back(i);
        }
    }
    
    pcl::copyPointCloud(*rtObstacles, inliers, *rtObstacles);
}

inline void SVMPathPlanning::generateRNG(const PointCloudType::Ptr & pathNodes, vector<Node> & nodeList, 
                                         const PointCloudType::Ptr & currentMap, const bool & doExtendedGraph,
                                         const bool & doSegmentChecking) {
    cout << __FUNCTION__ << ":" << __LINE__ << endl;
    if (pathNodes->size() == 0)
        return;
    
    cout << __FUNCTION__ << ":" << __LINE__ << endl;
    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;
    
    cout << __FUNCTION__ << ":" << __LINE__ << endl;
    cout << pathNodes->size() << endl;
    vector< pair<uint32_t, uint32_t> > edges;
    edges.reserve(pathNodes->size() * pathNodes->size());
    
    cout << __FUNCTION__ << ":" << __LINE__ << endl;
    if (doExtendedGraph) {
        
        cout << __FUNCTION__ << ":" << __LINE__ << endl;
        // Creating the KdTree object for the search method of the extraction
        pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
        tree->setInputCloud (pathNodes);
        
        cout << __FUNCTION__ << ":" << __LINE__ << endl;
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<PointType> ec;
        ec.setClusterTolerance (m_distBetweenSamples);
        ec.setMinClusterSize (1);
        ec.setMaxClusterSize (INT_MAX);
        ec.setSearchMethod (tree);
        ec.setInputCloud (pathNodes);
        ec.extract (cluster_indices);

        cout << __FUNCTION__ << ":" << __LINE__ << endl;
        vector<uint32_t> nodeLabels;
        nodeLabels.resize(pathNodes->size());
        uint32_t label = 0;
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); it++, label++) {
            for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end (); pit++) {
                nodeLabels[*pit] = label;
            }
        }
                
        cout << __FUNCTION__ << ":" << __LINE__ << endl;
        PGPUDTPARAMS pInput = new GPUDTPARAMS;
        
        pInput->minX = m_minCorner.x;
        pInput->minY = m_minCorner.y;
        pInput->maxX = m_maxCorner.x;
        pInput->maxY = m_maxCorner.y;
        
        pInput->nPoints = pathNodes->size();
        pInput->points = new gpudtVertex[pInput->nPoints];
        
        pInput->nConstraints = 0;
        
        cout << __FUNCTION__ << ":" << __LINE__ << endl;
        uint32_t idx = 0;
        for (PointCloudType::iterator it = pathNodes->begin(); it != pathNodes->end(); it++, idx++) {
            pInput->points[idx].x = it->x;
            pInput->points[idx].y = it->y;
        }
        pInput->fboSize = FBO_SIZE;                    // Texture size to be used (256, 512, 1024, 2048, 4096)
        
        // Run GPU-DT
        
        clock_t tv[2];
        
        PGPUDTOUTPUT pOutput = NULL;
        
        tv[0] = clock();
        
        cout << __FUNCTION__ << ":" << __LINE__ << endl;
        try {
            pOutput = gpudtComputeDT(pInput);
        } catch (...) {
            cerr << "Problem when launching GPUDT" << endl;
            pOutput = NULL;
        }
        
        cout << __FUNCTION__ << ":" << __LINE__ << endl;
        tv[1] = clock();
        
        printf("GPU-DT time: %.4fs\n", (tv[1]-tv[0])/(REAL)CLOCKS_PER_SEC);      
        
        cout << __FUNCTION__ << ":" << __LINE__ << endl;
        if (pOutput) {
            for (uint32_t i = 0; i < pOutput->nTris; i++) {
                const gpudtTriangle & triangle = pOutput->triangles[i];
                
                for (unsigned j = 0; j < 3; j++) {
                    const int & idx1 = triangle.vtx[j];
                    const int & idx2 = triangle.vtx[(j + 1) % 3];
                    if (nodeLabels[idx1] != nodeLabels[idx2]) {
                        edges.push_back(make_pair<uint32_t, uint32_t>(idx1, idx2));
                    }
                }
            }
            
            gpudtReleaseDTOutput(pOutput);
        }
        cout << __FUNCTION__ << ":" << __LINE__ << endl;
        
        delete pInput->points;
        delete pInput;
        delete pOutput;
        cout << __FUNCTION__ << ":" << __LINE__ << endl;
    }
    
    cout << __FUNCTION__ << ":" << __LINE__ << endl;
    // Graph is completed with lines that not form a polygon
    pcl::search::KdTree<PointType>::Ptr treeNNG (new pcl::search::KdTree<PointType>);
    treeNNG->setInputCloud (pathNodes);
    treeNNG->setSortedResults(true);
    
    cout << __FUNCTION__ << ":" << __LINE__ << endl;
    for (uint32_t i = 0; i < pathNodes->size(); i++) {
        treeNNG->radiusSearch(pathNodes->at(i), m_distBetweenSamples, pointIdxNKNSearch, pointNKNSquaredDistance);
        
        cout << __FUNCTION__ << ":" << __LINE__ << endl;
        for (uint32_t j = 0; j < pointIdxNKNSearch.size(); j++) {
            edges.push_back(make_pair<uint32_t, uint32_t>(i, pointIdxNKNSearch[j]));
        }
    }
    cout << __FUNCTION__ << ":" << __LINE__ << endl;
    
    checkSegments(pathNodes, nodeList, currentMap, edges, doSegmentChecking);
    cout << __FUNCTION__ << ":" << __LINE__ << endl;
}

void SVMPathPlanning::checkSegments(const PointCloudType::Ptr & pathNodes, vector<Node> & nodeList, 
                                    const PointCloudType::Ptr & currentMap, const vector< pair<uint32_t, uint32_t> > & edges,
                                    const bool & doSegmentChecking)
{
    float2 * pointsInMap = new float2[currentMap->size()];
    float2 * edgeU = new float2[edges.size()];
    float2 * edgeV = new float2[edges.size()];
    bool * validEdges =  new bool[edges.size()];
    
    uint32_t i = 0;
    for (PointCloudType::iterator it = currentMap->begin(); it != currentMap->end(); it++, i++) {
        pointsInMap[i].x = it->x;
        pointsInMap[i].y = it->y;
    }
    
    i = 0;
    for (vector< pair<uint32_t, uint32_t> >::const_iterator it = edges.begin(); it != edges.end(); it++, i++) {
        edgeU[i].x = pathNodes->at(it->first).x;
        edgeU[i].y = pathNodes->at(it->first).y;

        edgeV[i].x = pathNodes->at(it->second).x;
        edgeV[i].y = pathNodes->at(it->second).y;
    }
    
    launchCheckEdges((const float2 *&)pointsInMap, currentMap->size(), (const float2 *&)edgeU, 
                     (const float2 *&)edgeV, edges.size(), (const float &)(m_carWidth / 2 + m_minDistCarObstacle), (bool *&)validEdges);
    
    for (uint32_t i = 0; i < edges.size(); i++) {
        if (validEdges[i] == true) {
            const Node & node1 = nodeList[edges[i].first];
            const Node & node2 = nodeList[edges[i].second];
            
            const PointType & point1 = (*m_nodeMap)[node1];
            const PointType & point2 = (*m_nodeMap)[node2];
    
            Edge edge = m_graph.addEdge(node1, node2);
                                        
            (*m_distMap)[edge] = pcl::euclideanDistance(point1, point2);
        }
    }
    
    delete pointsInMap;
    delete edgeU;
    delete edgeV;
    delete validEdges;
}

void SVMPathPlanning::getGraph(PointCloudType::Ptr & pointCloud) {
    pointCloud = PointCloudType::Ptr(new PointCloudType);  
    
    PointCloudTypeExt::Ptr pointCloudRGB(new PointCloudTypeExt);
    
    for (Graph::ArcIt it(m_graph); it != lemon::INVALID; ++it) {
        Edge currentEdge;
        addLineToPointCloud((*m_nodeMap)[m_graph.u(it)], (*m_nodeMap)[m_graph.v(it)], 255, 0, 0, pointCloudRGB);
    }
        
    pcl::copyPointCloud(*pointCloudRGB, *pointCloud);
}
