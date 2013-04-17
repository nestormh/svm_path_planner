/*
    Copyright (c) 2013, Néstor Morales Hernández <nestor@isaatc.ull.es>
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the <organization> nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY NÃ©stor Morales HernÃ¡ndez <nestor@isaatc.ull.es> ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL NÃ©stor Morales HernÃ¡ndez <nestor@isaatc.ull.es> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "svmpathplanning.h"

#include <time.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <vector_functions.h>

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
#include <pcl/surface/gp3.h>

#include <boost/property_map/property_map.hpp>

#include <lemon/dijkstra.h>
#include <lemon/connectivity.h>

#include <../gpudt/gpudt.h>

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

void SVMPathPlanning::loadDataFromFile (const std::string & fileName,
                                        PointCloudType::Ptr & X,
                                        PointCloudType::Ptr & Y )
{
    
    X = PointCloudType::Ptr ( new PointCloudType );
    Y = PointCloudType::Ptr ( new PointCloudType );
    
    ifstream fin ( fileName.c_str(), ios::in );
    fin.ignore ( 1024, '\n' );
    
    uint32_t idx = 0;
    while ( ! fin.eof() ) {
        int32_t classType;
        double x, y;
        string field;
        vector<string> tokens;
        
        fin >> classType;
        fin >> field;
        
//         cout << classType << ", " << field << endl;
        
        boost::split ( tokens, field, boost::is_any_of ( ":" ) );
        
        std::stringstream ss1;
        ss1 << tokens[1];
        ss1 >> x;
        
        fin >> field;
        
        boost::split ( tokens, field, boost::is_any_of ( ":" ) );
        
        std::stringstream ss2;
        ss2 << tokens[1];
        ss2 >> y;
        
//         cout << classType << " (" << x << ", " << y << ")" << endl;
        
        PointType p;
        p.x = x;
        p.y = y;
//         p.z = p.r = p.g = p.b = 0.0;
        
//         if ( classType == 1 ) {
//             p.r = 255;
//             X->push_back ( p );
//         } else {
//             p.g = 255;
//             Y->push_back ( p );
//         }       
        fin.ignore ( 1024, '\n' );
    }
    
    fin.close();
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
    
    classes.reserve(cluster_indices.size());
    
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
                                          PointCloudTypeExt::Ptr & linesPointCloud, double zOffset = 0.0) {
    
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

void SVMPathPlanning::visualizeClasses(const vector< PointCloudType::Ptr > & classes, const PointCloudType::Ptr & pathNodes,
                                       const PointCloudType::Ptr & rtObstacles = PointCloudType::Ptr(),
                                       const PointCloudType::Ptr & path = PointCloudType::Ptr()) {
    
    PointCloudTypeExt::Ptr pointCloud(new PointCloudTypeExt);
    
    for (uint32_t i = 0; i < classes.size(); i++) {
        PointCloudType::Ptr currentClass = classes[i];
        
        uchar color[] = { rand() & 255, rand() & 255, rand() & 255 };
        
        for (uint32_t j = 0; j < currentClass->size(); j++) {
            
            PointTypeExt point;
            
            point.x = currentClass->at(j).x;
            point.y = currentClass->at(j).y;
            point.z = 0.0;
            point.r = color[0];
            point.g = color[1];
            point.b = color[2];
            
            pointCloud->push_back(point);
        }
    }
    
    PointCloudTypeExt::Ptr trajectory(new PointCloudTypeExt);
    trajectory->reserve(pathNodes->size());
    
    for (uint32_t i = 0; i < pathNodes->size(); i++) {
        PointTypeExt point;
        
        point.x = pathNodes->at(i).x;
        point.y = pathNodes->at(i).y;
        point.z = -1.0;
        point.r = 0;
        point.g = 255;
        point.b = 0;
        
        trajectory->push_back(point);
    }

//     PointCloudTypeExt::Ptr footprintStartRGB(new PointCloudTypeExt);
//     footprintStartRGB->reserve(footprintStart->size());
//     for (PointCloudType::iterator it = footprintStart->begin(); it != footprintStart->end(); it++) {
//         PointTypeExt point; 
//         
//         point.x = it->x;
//         point.y = it->y;
//         point.z = -2.0;
//         point.r = 128;
//         point.g = 255;
//         point.b = 128;
//         
//         footprintStartRGB->push_back(point);
//     }
//     
//     PointCloudTypeExt::Ptr footprintGoalRGB(new PointCloudTypeExt);
//     footprintGoalRGB->reserve(footprintGoal->size());
//     for (PointCloudType::iterator it = footprintGoal->begin(); it != footprintGoal->end(); it++) {
//         PointTypeExt point; 
//         
//         point.x = it->x;
//         point.y = it->y;
//         point.z = -2.0;
//         point.r = 128;
//         point.g = 255;
//         point.b = 128;
//         
//         footprintGoalRGB->push_back(point);
//     }
    
    PointCloudTypeExt::Ptr pathPointCloud(new PointCloudTypeExt);
    if (path.get() != NULL) {
        PointType lastPoint;
        for (uint32_t i = 0; i < path->size(); i++) {
            const PointType & point = path->at(i);
            
            if (i != 0)
                addLineToPointCloud(lastPoint, point, 0, 255, 255, pathPointCloud, 1.0);
            lastPoint = point;
            
    //         cout << point << endl;
        }
    }
    
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->initCameraParameters();
    viewer->addCoordinateSystem();
    
    pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgb(pointCloud);
    viewer->addPointCloud<PointTypeExt> (pointCloud, rgb, "pointCloud");
    
    pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbTrajectory(trajectory);
    viewer->addPointCloud<PointTypeExt> (trajectory, rgbTrajectory, "trajectory");
    
//     pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbFootprintStart(footprintStart);
//     viewer->addPointCloud<PointTypeExt> (footprintStart, rgbFootprintStart, "footprintStart");
//     
//     pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbFootprintGoal(footprintGoal);
//     viewer->addPointCloud<PointTypeExt> (footprintGoal, rgbFootprintGoal, "footprintGoal");
    
    if (path.get() != NULL) {
        pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbPath(pathPointCloud);
        viewer->addPointCloud<PointTypeExt> (pathPointCloud, rgbPath, "pathPointCloud");
    }
    
    PointCloudTypeExt::Ptr graphPointCloud(new PointCloudTypeExt);  
    for (Graph::ArcIt it(m_graph); it != lemon::INVALID; ++it) {
        Edge currentEdge;
        addLineToPointCloud((*m_nodeMap)[m_graph.u(it)], (*m_nodeMap)[m_graph.v(it)], 255, 0, 0, graphPointCloud);
    }
        
    pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbTriangulation(graphPointCloud);
    viewer->addPointCloud<PointTypeExt> (graphPointCloud, rgbTriangulation, "graphPointCloud");
        
    while (! viewer->wasStopped ()) {    
        viewer->spinOnce();       
    }
}

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
        
    CornerLimitsType minCorner, maxCorner, interval;
    clusterize(inputCloud, m_classes, minCorner, maxCorner);
    
    vector< PointCloudType::Ptr >::iterator it1, it2;
    
    interval = make_double2(maxCorner.x - minCorner.x, 
                            maxCorner.y - minCorner.y);
    m_minCorner = minCorner;
    m_maxCorner = maxCorner;
    
    m_distBetweenSamples = 1.5 * sqrt((interval.x / m_mapGridSize.width) * (interval.x / m_mapGridSize.width) +
                                      (interval.y / m_mapGridSize.height) * (interval.y / m_mapGridSize.height));
    
    m_originalMap = PointCloudType::Ptr(new PointCloudType);
    
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
    generateRNG(m_pathNodes, m_nodeList, m_originalMap);
    
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    std::cout << "Total time for graph generation = " << elapsed << endl;
    
    m_mapGenerated = true;
    
    if (visualize) {
        visualizeClasses(m_classes, m_pathNodes);
    }
}

bool SVMPathPlanning::findShortestPath(const PointType& start, const double & startOrientation,
                                       const PointType & goal, const double & goalOrientation,
                                       PointCloudType::Ptr rtObstacles, bool visualize = false)
{
    
    for (Graph::EdgeIt it(m_graph); it != lemon::INVALID; ++it) {
        Edge currentEdge = it;
        m_graph.erase(currentEdge);
    }
    
    struct timespec startTime, finishTime;
    double elapsed;
    
    clock_gettime(CLOCK_MONOTONIC, &startTime);
    
    CornerLimitsType minCornerRT;
    CornerLimitsType maxCornerRT;
    
    vector< PointCloudType::Ptr > classes;
    
    filterExistingObstacles(rtObstacles);
    
    CornerLimitsType dummyCorner1, dummyCorner2;
    clusterize(rtObstacles, classes, dummyCorner1, dummyCorner2);
    
    PointCloudType::Ptr currentMap(new PointCloudType);
    *currentMap += *m_originalMap;
    *currentMap += *rtObstacles;    

    PointCloudType::Ptr footprintStart, footprintGoal;
    bool startCheck = getFootPrint(start, currentMap, footprintStart);
    bool goalCheck = getFootPrint(goal, currentMap, footprintGoal);
    
    if (! startCheck) {
        cerr << "Failed to find a path: Current position is too near to an obstacle or colliding with it" << endl;
        
        return false;
    }
    if (! startCheck) {
        cerr << "Failed to find a path: Goal position is not clear" << endl;
        
        return false;
    }
//     classes.clear();
    classes.push_back(footprintStart);
    classes.push_back(footprintGoal);
    
    PointCloudType::Ptr pathNodesRT(new PointCloudType);
    pcl::copyPointCloud(*m_pathNodes, *pathNodesRT);
   
    vector<Node> nodeListRT;
    copy(m_nodeList.begin(), m_nodeList.end(), back_inserter(nodeListRT));
    int maxNodeId = nodeListRT.size();
    
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
    
    generateRNG(pathNodesRT, nodeListRT, currentMap);
    
    clock_gettime(CLOCK_MONOTONIC, &finishTime);
    elapsed = (finishTime.tv_sec - startTime.tv_sec);
    elapsed += (finishTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;
    
    std::cout << "Total time for graph generation in real time = " << elapsed << endl;
    
//     copy(m_classes.begin(), m_classes.end(), back_inserter(classes));    
//     visualizeClasses(classes, pathNodesRT, rtObstacles);
    
//     return true;
    
    double distStart = DBL_MAX, distGoal = DBL_MAX;
    Node startNode, goalNode;
    PointType tmpStartPoint(start.x + m_carWidth * cos(startOrientation),
                            start.y + m_carWidth * sin(startOrientation),
                            0.0);
    PointType tmpGoalPoint(goal.x + m_carWidth * cos(goalOrientation + M_PI),
                           goal.y + m_carWidth * sin(goalOrientation + M_PI),
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
    
    Node currentNode = startNode;
    m_path->push_back(start);
    while (currentNode != goalNode) {
        currentNode = dijkstra.predNode(currentNode);
        m_path->push_back((*m_nodeMap)[currentNode]);
    }
    m_path->push_back(goal);
        
    if (visualize) {
        copy(m_classes.begin(), m_classes.end(), back_inserter(classes));
        visualizeClasses(classes, m_pathNodes, rtObstacles, m_path);
    }
    
    return true;
}

bool SVMPathPlanning::getFootPrint(const PointType & position, const PointCloudType::Ptr & currentMap, 
                                   PointCloudType::Ptr & footprint) {
    
    vector<int> idxMap, idxRT;
    vector<float> distMap, distRT;
    
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<PointType>::Ptr treeMap (new pcl::search::KdTree<PointType>);
    treeMap->setInputCloud (currentMap);
    
//     treeMap->nearestKSearch(position, 1, idxMap, distMap);
//     double radius = max((double)sqrt(distMap[0]) - m_minDistBetweenObstacles, m_carWidth);
    
    treeMap->radiusSearch(position, (m_carWidth / 2 + m_minDistCarObstacle), idxMap, distMap);
    
    // If idxMap != 0, we are colliding with an obstacle
    if (idxMap.size() != 0) {
        return false;
    }
    
    footprint = PointCloudType::Ptr(new PointCloudType);
    footprint->reserve(36);
    
    for (double alpha = 0.0; alpha < 2 * M_PI; alpha += M_PI / 180 * 10) {
        const PointType point(m_carWidth * cos(alpha) + position.x, 
                              m_carWidth * sin(alpha) + position.y,
                              0.0);
        
        treeMap->radiusSearch(point, (m_carWidth / 2 + m_minDistCarObstacle), idxMap, distMap);
        
        // If idxMap != 0, we are colliding with an obstacle
        if (idxMap.size() != 0) {
            return false;
        }
        
        footprint->push_back(point);
    }
    
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

inline void SVMPathPlanning::generateRNG(const PointCloudType::Ptr & pathNodes, vector<Node> & nodeList, const PointCloudType::Ptr & currentMap) {
    
    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;
    
    vector< pair<uint32_t, uint32_t> > edges;
    edges.reserve(pathNodes->size() * pathNodes->size());
    
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
    tree->setInputCloud (pathNodes);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointType> ec;
    ec.setClusterTolerance (m_distBetweenSamples);
    ec.setMinClusterSize (1);
    ec.setMaxClusterSize (INT_MAX);
    ec.setSearchMethod (tree);
    ec.setInputCloud (pathNodes);
    ec.extract (cluster_indices);

//     m_graph.clear();
    vector<uint32_t> nodeLabels;
    nodeLabels.resize(pathNodes->size());
    uint32_t label = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); it++, label++) {
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end (); pit++) {
//             Node node = m_graph.addNode();
//             (*m_nodeMap)[node] = pathNodes->at(*pit);
//             nodeList.push_back(node);
            nodeLabels[*pit] = label;
        }
    }
            
    PGPUDTPARAMS pInput = new GPUDTPARAMS;
    
    pInput->minX = m_minCorner.x;
    pInput->minY = m_minCorner.y;
    pInput->maxX = m_maxCorner.x;
    pInput->maxY = m_maxCorner.y;
    
    pInput->nPoints = pathNodes->size();
    pInput->points = new gpudtVertex[pInput->nPoints];
    
    pInput->nConstraints = 0;
    
    uint32_t idx = 0;
    for (PointCloudType::iterator it = pathNodes->begin(); it != pathNodes->end(); it++, idx++) {
        pInput->points[idx].x = it->x;
        pInput->points[idx].y = it->y;
    }
    pInput->fboSize = 256;                    // Texture size to be used (256, 512, 1024, 2048, 4096)
    
    // Run GPU-DT
    
    clock_t tv[2];
    
    PGPUDTOUTPUT pOutput = NULL;
    
    tv[0] = clock();
    
    pOutput = gpudtComputeDT(pInput);
    
    tv[1] = clock();
    
    printf("GPU-DT time: %.4fs\n", (tv[1]-tv[0])/(REAL)CLOCKS_PER_SEC);      
    
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
    
    if (pOutput)
        gpudtReleaseDTOutput(pOutput);
    
    delete pInput->points;
    delete pInput;
    delete pOutput;
    
    // Graph is completed with lines that not form a polygon
    pcl::KdTreeFLANN<PointType>::Ptr treeNNG (new pcl::KdTreeFLANN<PointType>);
    treeNNG->setInputCloud (pathNodes);
    treeNNG->setSortedResults(true);
    
    for (uint32_t i = 0; i < pathNodes->size(); i++) {
        treeNNG->radiusSearch(pathNodes->at(i), m_distBetweenSamples, pointIdxNKNSearch, pointNKNSquaredDistance);
//         treeNNG->radiusSearch(pathNodes->at(i), 1.1 * (m_minDistCarObstacle + m_carWidth), pointIdxNKNSearch, pointNKNSquaredDistance);
        
        for (uint32_t j = 0; j < pointIdxNKNSearch.size(); j++) {
            edges.push_back(make_pair<uint32_t, uint32_t>(i, pointIdxNKNSearch[j]));
        }
    }
    
    checkSegments(pathNodes, nodeList, currentMap, edges);
}

void SVMPathPlanning::checkSegments(const PointCloudType::Ptr & pathNodes, vector<Node> & nodeList, 
                                    const PointCloudType::Ptr & currentMap, const vector< pair<uint32_t, uint32_t> > & edges)
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
