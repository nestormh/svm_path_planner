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


#include "svmpathplanningsingle.h"

#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>

#include <vector_functions.h>

#include <lemon/connectivity.h>
#include <lemon/dijkstra.h>

using namespace svmpp;

SVMPathPlanningSingle::SVMPathPlanningSingle() : SVMPathPlanning()
{
    m_Np = 20;
    m_maxIterations = 100;
//     m_Dv = m_carWidth = 2.5;
    m_Dv = 5;
    m_minDistCarObstacle = 0.0000001;
//     m_carWidth = 0.5;
    
    // default values
    m_param.svm_type = C_SVC;
    m_param.kernel_type = RBF;
    m_param.degree = 3;
    m_param.gamma = 10;//300;        // 1/num_features
    m_param.coef0 = 0;
    m_param.nu = 0.5;
    m_param.cache_size = 100;
    m_param.C = 1000;//500;
    m_param.eps = 1e-2;
    m_param.p = 0.1;
    m_param.shrinking = 0;
    m_param.probability = 0;
    m_param.nr_weight = 0;
    m_param.weight_label = NULL;
    m_param.weight = NULL;
    //     cross_validation = 0;
    
    m_originalMap = PointCloudType::Ptr(new PointCloudType);
}

void SVMPathPlanningSingle::setMap(const pcl::PointCloud< PointType >::Ptr& inputCloud)
{
    m_originalMap = PointCloudType::Ptr(new PointCloudType);
    *m_originalMap = *inputCloud;
}

inline bool SVMPathPlanningSingle::isLeftFromLine(const PointType & line1, const PointType & line2, const PointType & point)
{
    return ((line2.x - line1.x)*(point.y - line1.y) - (line2.y - line1.y)*(point.x - line1.x)) > 0;
}

bool SVMPathPlanningSingle::findShortestPath(const PointType& start, const double & startOrientation,
                                            const PointType & goal, const double & goalOrientation,
                                            const PointCloudType::Ptr & rtObstacles, bool visualize) 
{
    *rtObstacles += *m_originalMap;
    
    vector<PointCloudType::Ptr> footprintStart, footprintGoal;
    bool startCheck = getFootPrint(start, startOrientation, rtObstacles, footprintStart);
    bool goalCheck = getFootPrint(goal, goalOrientation, rtObstacles, footprintGoal);
    
    if (! startCheck) {
        cerr << "Failed to find a path: Current position is too near to an obstacle or colliding with it" << endl;
        
        return false;
    }
    if (! startCheck) {
        cerr << "Failed to find a path: Goal position is not clear" << endl;
        
        return false;
    }
        
    PointType p1 = start;
    PointType p2 = goal;

    double mapOrientation = atan2(goal.y - start.y, goal.x - start.x);
    
    Eigen::Affine3f tMatrix; 
//     pcl::getTransformation (start.x, start.y, 0.0, 0.0, 0.0, -mapOrientation, tMatrix);
    pcl::getTransformation (0.0, 0.0, 0.0, 0.0, 0.0, -mapOrientation, tMatrix);
    pcl::transformPointCloud(*rtObstacles, *rtObstacles, tMatrix);
    p1 = pcl::transformPoint(p1, tMatrix);
    p2 = pcl::transformPoint(p2, tMatrix);
    pcl::transformPointCloud(*footprintStart[0], *footprintStart[0], tMatrix);
    pcl::transformPointCloud(*footprintStart[1], *footprintStart[1], tMatrix);
    pcl::transformPointCloud(*footprintGoal[0], *footprintGoal[0], tMatrix);
    pcl::transformPointCloud(*footprintGoal[1], *footprintGoal[1], tMatrix);
    
    vector< PointCloudType::Ptr > classes;
    CornerLimitsType dummyCorner1, dummyCorner2;
    clusterize(rtObstacles, classes, dummyCorner1, dummyCorner2);
        
//     vector < vector <bool> > visitedPatterns;
    
    bool reached = false;
    
    Eigen::Vector4f centroid;
    vector <bool> currentPattern;
    vector <bool> fixedClasses;
    currentPattern.resize(classes.size());
    fixedClasses.resize(classes.size());
    uint32_t idx = 0;
    for (vector< PointCloudType::Ptr >::iterator it = classes.begin(); it != classes.end(); it++, idx++) {
        pcl::compute3DCentroid(*(*it), centroid);
        const PointType pointCentroid(centroid[0], centroid[1], centroid[2]);
        
        if (pointCentroid.y < p1.y) {
            currentPattern[idx] = true;
        } else {
            currentPattern[idx] = false;
        }
        
        fixedClasses[idx] = isFixedClass(*it, p1.y, currentPattern[idx]);
    }
    
    vector < vector <bool> > availablePatterns;
    availablePatterns.push_back(currentPattern);
    
    getAvailablePatterns(currentPattern, fixedClasses, availablePatterns, 0);
    
    const uint32_t maxIterations = std::min((uint32_t)availablePatterns.size(), m_Np + 1);
    
    PointCloudType::Ptr path(new PointCloudType);
    for (uint32_t i = 0; (i < maxIterations)/* && (! reached)*/; i++) {
        const vector <bool> & currentPattern = availablePatterns[i];
        
        path->clear();
        
        double cost = DBL_MAX, minCost = DBL_MAX;
        if (getPath(currentPattern, classes, footprintStart, footprintGoal, p1, p2, path, cost, visualize)) {
            reached = true;
            if (cost < minCost) {
                cout << cost << " < " << minCost << endl;
                minCost = cost;
                pcl::copyPointCloud(*path, *m_path);
            }
        }
    }
    
//     visitedPatterns.push_back(currentPattern);
//     
//     double minCost = DBL_MAX;
//     reached = getPath(currentPattern, classes, footprintStart, footprintGoal, p1, p2, m_path, minCost, visualize);
//     
//     
// //     visualizeClasses(classes, m_pathNodes, PointCloudType::Ptr(), m_path);
//     
// //     exit(0);
//     
//     PointCloudType::Ptr path(new PointCloudType);
//     uint32_t totalIterations = 0;
//     while ((! reached) || (totalIterations < m_Np)) {
//         totalIterations++;
//         
//         if (totalIterations > m_maxIterations) {
//             cout << "Maximal number of iterations reached. Finishing..." << endl;
//             
//             break;
//         }
//         
//         cout << "Iteration: " << totalIterations << " / " << m_Np << endl;
//         
//         if (! getNewPattern(visitedPatterns, fixedClasses)) {
//             cout << "Maximal number of patterns tested. Finishing...." << endl;
//             break;
//         }
//         
//         const vector <bool> & currentPattern = visitedPatterns[visitedPatterns.size() - 1];
//         
//         path->clear();
//         
//         double cost = DBL_MAX;
//         if (getPath(currentPattern, classes, footprintStart, footprintGoal, p1, p2, path, cost, visualize)) {
//             reached = true;
//             if (cost < minCost) {
//                 cout << cost << " < " << minCost << endl;
//                 minCost = cost;
//                 pcl::copyPointCloud(*path, *m_path);
//             }
//         }
//     }
    
    if (! reached) {
        cerr << "Unable to find a path." << endl;
    }
    
    Eigen::Affine3f inverseTMatrix;
//     pcl::getTransformation (-start.x, -start.y, 0.0, 0.0, 0.0, mapOrientation, inverseTMatrix);
    pcl::getTransformation (0.0, 0.0, 0.0, 0.0, 0.0, mapOrientation, inverseTMatrix);
    
    pcl::transformPointCloud(*m_path, *m_path, inverseTMatrix);
 
    pcl::transformPointCloud(*rtObstacles, *rtObstacles, inverseTMatrix);
    p1 = pcl::transformPoint(p1, inverseTMatrix);
    p2 = pcl::transformPoint(p2, inverseTMatrix);
    pcl::transformPointCloud(*footprintStart[0], *footprintStart[0], inverseTMatrix);
    pcl::transformPointCloud(*footprintStart[1], *footprintStart[1], inverseTMatrix);
    pcl::transformPointCloud(*footprintGoal[0], *footprintGoal[0], inverseTMatrix);
    pcl::transformPointCloud(*footprintGoal[1], *footprintGoal[1], inverseTMatrix);
    
    pcl::transformPointCloud(*m_pathNodes, *m_pathNodes, inverseTMatrix);
    
    for (Graph::NodeIt it(m_graph); it != lemon::INVALID; ++it) {
        const Node currentNode = it;
        
        PointType & point = (*m_nodeMap)[currentNode];
        
        point = pcl::transformPoint(point, inverseTMatrix);
    }
        
    
//     if (visualize) {
//         visualizeClasses(classes, m_pathNodes, PointCloudType::Ptr(), m_path);
//     }    
    
    return reached;
}

// void SVMPathPlanningSingle::visualizer(const PointCloudType::Ptr & class1, const PointCloudType::Ptr & class2, const PointType & start, const PointType & goal) 
// {
//     
//     PointCloudTypeExt::Ptr pointCloudClass1(new PointCloudTypeExt);
//     PointCloudTypeExt::Ptr pointCloudClass2(new PointCloudTypeExt);
//     
//     for (uint32_t i = 0; i < class1->size(); i++) {
//         PointTypeExt point;
//         point.x = class1->at(i).x;
//         point.y = class1->at(i).y;
//         point.z = 0.0;
//         point.r = 255;
//         point.g = 0;
//         point.b = 0;
//         
//         pointCloudClass1->push_back(point);
//     }
//     for (uint32_t i = 0; i < class2->size(); i++) {
//         PointTypeExt point;
//         point.x = class2->at(i).x;
//         point.y = class2->at(i).y;
//         point.z = 0.0;
//         point.r = 0;
//         point.g = 0;
//         point.b = 255;
//         
//         pointCloudClass2->push_back(point);
//     }
//     
//     boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//     viewer->setBackgroundColor (0, 0, 0);
//     viewer->initCameraParameters();
//     viewer->addCoordinateSystem();
//     
//     pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> pointCloudClass1RGB(pointCloudClass1);
//     viewer->addPointCloud<PointTypeExt> (pointCloudClass1, pointCloudClass1RGB, "pointCloudClass1");
//     
//     pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> pointCloudClass2RGB(pointCloudClass2);
//     viewer->addPointCloud<PointTypeExt> (pointCloudClass2, pointCloudClass2RGB, "pointCloudClass2");
//     
//     viewer->addLine(start, goal, 0, 255, 0, "Line");
//     
//     while (! viewer->wasStopped ()) {    
//         viewer->spinOnce();       
//     }
// }

bool SVMPathPlanningSingle::getNewPattern(vector< std::vector< bool > >& visitedPatterns, const vector<bool> & fixedClasses)
{
    vector<bool> fliped;
    fliped.resize(visitedPatterns[0].size());
    vector<bool> currentPattern;
    currentPattern.resize(visitedPatterns[0].size());
    for (uint32_t i = 0; i < fliped.size(); i++) {
        if (fixedClasses[i])
            fliped[i] = true;
        else
            fliped[i] = false;
        currentPattern[i] = visitedPatterns[visitedPatterns.size() - 1][i];
    }
    for (uint32_t i = 0; i < fliped.size(); i++) {
        uint32_t pos = rand() % fliped.size();
        while (fliped[pos]) {
            pos = rand() % fliped.size();
        }
        fliped[pos] = true;
        
        currentPattern[pos] = !currentPattern[pos];
        
        for (uint32_t j = 0; j < visitedPatterns.size(); j++) {
            for (uint32_t k = 0; k < visitedPatterns[j].size(); k++) {
                if (visitedPatterns[j][k] != currentPattern[k]) {
                    visitedPatterns.push_back(currentPattern);
                    return true;
                }
            }
        }
        
        currentPattern[pos] = !currentPattern[pos];
    }

    return false;
}
inline bool SVMPathPlanningSingle::getPath(const vector <bool> & currentPattern,
                                          const vector< PointCloudType::Ptr > & classes,
                                          const vector<PointCloudType::Ptr> & footprintStart, 
                                          const vector<PointCloudType::Ptr> & footprintGoal,
                                          const PointType & p1, const PointType & p2,
                                          PointCloudType::Ptr & path, double & cost,
                                          const bool & visualize)
{
 
    for (Graph::EdgeIt it(m_graph); it != lemon::INVALID; ++it) {
        Edge currentEdge = it;
        m_graph.erase(currentEdge);
    }
    m_nodeList.clear();
    
    cost = DBL_MAX;
    
//     double minX = DBL_MIN, maxX = DBL_MAX;
//     for (uint32_t i = 0; i < 2; i++) {
//         for (uint32_t j = 0; j < footprintStart[i]->size(); j++) {
//             if (footprintStart[i]->at(j).x > minX) minX = footprintStart[i]->at(j).x;
//             if (footprintGoal[i]->at(j).x < maxX) maxX = footprintGoal[i]->at(j).x;
//         }
//     }
    
    double minX = p1.x + m_carWidth;
    double maxX = p2.x - m_carWidth;
    
    PointCloudType::Ptr positiveGuide(new PointCloudType);
    PointCloudType::Ptr negativeGuide(new PointCloudType);
    
    positiveGuide->resize(((maxX - minX) / m_minPointDistance) + 1);
    negativeGuide->resize(((maxX - minX) / m_minPointDistance) + 1);
    
    for (uint32_t i = 0; i < positiveGuide->size(); i++) {
        //         positiveGuide->at(i) = PointType(minX + i * m_minPointDistance, p1.y + m_Dv, 0.0);
        //         negativeGuide->at(i) = PointType(minX + i * m_minPointDistance, p1.y - m_Dv, 0.0);
        positiveGuide->at(i) = PointType(minX + i * m_minPointDistance, p1.y, 0.0);
        negativeGuide->at(i) = PointType(minX + i * m_minPointDistance, p1.y, 0.0);
    }
    
    PointCloudType::Ptr class1(new PointCloudType);
    PointCloudType::Ptr class2(new PointCloudType);

    uint32_t idx = 0;
    for (vector< PointCloudType::Ptr >::const_iterator it = classes.begin(); it != classes.end(); it++, idx++) {
        if (currentPattern[idx]) {
            *class1 += *(*it);
        } else {
            *class2 += *(*it);
        }
        for (PointCloudType::iterator it2 = (*it)->begin(); it2 != (*it)->end(); it2++) {
            if ((it2->x > minX) && (it2->x < maxX)) {
                if ((currentPattern[idx]) && (it2->y > p1.y)) {
                    const int pos = (negativeGuide->size() * (it2->x - minX) / (maxX - minX));
                    
                    const int minPos = max(0, pos - 1);
                    const int maxPos = min(pos + 1, (int)positiveGuide->size() - 1);
                    
                    for (int32_t i = minPos; i <= maxPos; i++) {
                        if (positiveGuide->at(i).y < it2->y) {
                            positiveGuide->at(i).y = it2->y;
                        }
                    }
                }
                if ((! currentPattern[idx]) && (it2->y < p1.y)) {
                    const int pos = (negativeGuide->size() * (it2->x - minX) / (maxX - minX));
                    
                    const int minPos = max(0, pos - 1);
                    const int maxPos = min(pos + 1, (int)positiveGuide->size() - 1);
                    
                    for (int32_t i = minPos; i <= maxPos; i++) {
                        if (negativeGuide->at(i).y > it2->y) {
                            negativeGuide->at(i).y = it2->y;
                        }
                    }
                }
            }
        }
    }
    
    for (uint32_t i = 0; i < positiveGuide->size(); i++) {
        negativeGuide->at(i).y -= m_Dv;
        positiveGuide->at(i).y += m_Dv;
    }
    
    *class1 += *footprintStart[0];
    *class1 += *footprintGoal[0];
    
    *class2 += *footprintStart[1];
    *class2 += *footprintGoal[1];
    
    *class1 += *negativeGuide;
    *class2 += *positiveGuide;
    
    CornerLimitsType minCorner = make_double2(DBL_MAX, DBL_MAX);
    CornerLimitsType maxCorner = make_double2(DBL_MIN, DBL_MIN);
    for (PointCloudType::iterator it = class1->begin(); it != class1->end(); it++) {
        if (it->x < minCorner.x) minCorner.x = it->x;
        if (it->y < minCorner.y) minCorner.y = it->y;
        
        if (it->x > maxCorner.x) maxCorner.x = it->x;
        if (it->y > maxCorner.y) maxCorner.y = it->y;
    }
    for (PointCloudType::iterator it = class2->begin(); it != class2->end(); it++) {
        if (it->x < minCorner.x) minCorner.x = it->x;
        if (it->y < minCorner.y) minCorner.y = it->y;
        
        if (it->x > maxCorner.x) maxCorner.x = it->x;
        if (it->y > maxCorner.y) maxCorner.y = it->y;
    }
    
    CornerLimitsType interval = make_double2(maxCorner.x - minCorner.x,
                                             maxCorner.y - minCorner.y);
    
    m_distBetweenSamples = 1.5 * sqrt((interval.x / m_mapGridSize.width) * (interval.x / m_mapGridSize.width) +
                                      (interval.y / m_mapGridSize.height) * (interval.y / m_mapGridSize.height));
    
    PointCloudType::Ptr pathNodes(new PointCloudType);
    vector<Node> nodeList;
    
    getBorderFromPointClouds (class1, class2, minCorner, maxCorner, interval, m_mapGridSize, 0, pathNodes, nodeList);
    
    PointCloudType::Ptr currentMap(new PointCloudType);
    for (vector< PointCloudType::Ptr >::const_iterator it = classes.begin(); it != classes.end(); it++, idx++)
        *currentMap += *(*it);
    
    generateRNG(pathNodes, nodeList, currentMap, false, false);
    
    vector<int> idxMap, idxRT;
    vector<float> distMap, distRT;
    Node startNode, goalNode;
    
    if (pathNodes->size() == 0) {
        cerr << "Unable to find a SVM decision boundary" << endl;
        return false;
    }
        
    
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<PointType>::Ptr treeMap (new pcl::search::KdTree<PointType>);
    treeMap->setInputCloud (pathNodes);
    treeMap->setSortedResults(true);
    
//     treeMap->radiusSearch(p1, 8 * m_carWidth, idxMap, distMap);
    treeMap->nearestKSearch(p1, 1, idxMap, distMap);
    if (idxMap.size() < 1) {
        cerr << "Unable to find a starting node..." << endl;
        
//         if (visualize) {
//             vector< PointCloudType::Ptr > tmpClasses;
//             tmpClasses.reserve(classes.size());
//             copy(classes.begin(), classes.end(), back_inserter(tmpClasses));
//             
//             
//             tmpClasses.push_back(footprintStart[0]);
//             tmpClasses.push_back(footprintStart[1]);
//             tmpClasses.push_back(footprintGoal[0]);
//             tmpClasses.push_back(footprintGoal[1]);
//             
//             tmpClasses.push_back(positiveGuide);
//             tmpClasses.push_back(negativeGuide);
//             
//             visualizeClasses(tmpClasses, pathNodes, PointCloudType::Ptr(), PointCloudType::Ptr());
//         }
        
        return false;
    }
    startNode = nodeList[idxMap[0]];
    
//     treeMap->radiusSearch(p2, 8 * m_carWidth, idxMap, distMap);
    treeMap->nearestKSearch(p2, 1, idxMap, distMap);
    if (idxMap.size() < 1) {
        cerr << "Unable to find a goal node..." << endl;
        
//         if (visualize) {
//             vector< PointCloudType::Ptr > tmpClasses;
//             tmpClasses.reserve(classes.size());
//             copy(classes.begin(), classes.end(), back_inserter(tmpClasses));
//             
//             
//             tmpClasses.push_back(footprintStart[0]);
//             tmpClasses.push_back(footprintStart[1]);
//             tmpClasses.push_back(footprintGoal[0]);
//             tmpClasses.push_back(footprintGoal[1]);
//             
//             tmpClasses.push_back(positiveGuide);
//             tmpClasses.push_back(negativeGuide);
//             
//             visualizeClasses(tmpClasses, pathNodes, PointCloudType::Ptr(), PointCloudType::Ptr());
//         }
        
        return false;
    }
    goalNode = nodeList[idxMap[0]];
    
    if (! lemon::connected(m_graph)) {
        Graph::NodeMap<uint32_t> compMap(m_graph);
        
        lemon::connectedComponents(m_graph, compMap);
        
        if (compMap[startNode] != compMap[goalNode]) {
//             cerr << "Unable to find a path: It is impossible to go from " << p1 << 
//             " to " << p2 << " with this configuration." << endl;
            
//             if (visualize) {
//                 vector< PointCloudType::Ptr > tmpClasses;
//                 tmpClasses.reserve(classes.size());
//                 copy(classes.begin(), classes.end(), back_inserter(tmpClasses));
//                 
//                 
//                 tmpClasses.push_back(footprintStart[0]);
//                 tmpClasses.push_back(footprintStart[1]);
//                 tmpClasses.push_back(footprintGoal[0]);
//                 tmpClasses.push_back(footprintGoal[1]);
//                 
//                 tmpClasses.push_back(positiveGuide);
//                 tmpClasses.push_back(negativeGuide);
//                 
//                 visualizeClasses(tmpClasses, pathNodes, PointCloudType::Ptr(), PointCloudType::Ptr());
//             }
            
            return false; 
        }
    }
    
    lemon::Dijkstra<Graph, EdgeMap> dijkstra(m_graph, *m_distMap);
    dijkstra.init();
    dijkstra.addSource(goalNode);
    dijkstra.start();
    
    cout << "Path found!!!" << endl;
    
    cost = dijkstra.dist(startNode);
    
    Node currentNode = startNode;
    path->push_back(p1);
    while (currentNode != goalNode) {
        currentNode = dijkstra.predNode(currentNode);
        path->push_back((*m_nodeMap)[currentNode]);
    }
    m_path->push_back(p2);
    
    cout << "Path length = " << path->size() << endl;
    cout << "Path cost = " << cost << endl;
    
    //     if (visualize) {
//                 vector< PointCloudType::Ptr > tmpClasses;
//             tmpClasses.reserve(classes.size());
//             copy(classes.begin(), classes.end(), back_inserter(tmpClasses));
//             
//             
//             tmpClasses.push_back(footprintStart[0]);
//             tmpClasses.push_back(footprintStart[1]);
//             tmpClasses.push_back(footprintGoal[0]);
//             tmpClasses.push_back(footprintGoal[1]);
//             
//             tmpClasses.push_back(positiveGuide);
//             tmpClasses.push_back(negativeGuide);
//             
//             visualizeClasses(tmpClasses, pathNodes, PointCloudType::Ptr(), path);
    //     }
    
    return true;
}

bool SVMPathPlanningSingle::isFixedClass(const PointCloudType::Ptr & pointCloud, const double& threshold, const double& isLower)
{
    for (PointCloudType::iterator it = pointCloud->begin(); it != pointCloud->end(); it++) {
        
        if (isLower && it->y > threshold)
            return false;
        if (!isLower && it->y < threshold)
            return false;
    }
    
    return true;
}

void SVMPathPlanningSingle::getAvailablePatterns(const vector<bool> & currentPattern, const vector<bool> & fixedClasses, 
                                                vector <vector <bool> > & availablePatterns, const uint32_t & idx) {
                                              
    if (idx == currentPattern.size())
        return;
    
    if (fixedClasses[idx]) {
        getAvailablePatterns(currentPattern, fixedClasses, availablePatterns, idx + 1);
    } else {
        getAvailablePatterns(currentPattern, fixedClasses, availablePatterns, idx + 1);
        
        std::vector<bool> newPattern( currentPattern.size() );
        std::copy( currentPattern.begin(), currentPattern.end(), newPattern.begin() );
        
        newPattern[idx] = !newPattern[idx];
        
        availablePatterns.push_back(newPattern);

        
        getAvailablePatterns(newPattern, fixedClasses, availablePatterns, idx + 1);
    }
}