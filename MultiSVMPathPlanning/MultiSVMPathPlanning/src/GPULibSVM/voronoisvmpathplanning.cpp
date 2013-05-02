/*
    Copyright (c) 2013, NÃ©stor Morales HernÃ¡ndez <nestor@isaatc.ull.es>
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


#include "voronoisvmpathplanning.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree.h>

#include <vector_functions.h>

#include <lemon/connectivity.h>
#include <lemon/dijkstra.h>

using namespace svmpp;

VoronoiSVMPathPlanning::VoronoiSVMPathPlanning()
{
    // default values
    m_param.svm_type = C_SVC;
    m_param.kernel_type = RBF;
    m_param.degree = 3;
    m_param.gamma = 2; //150;//300;        // 1/num_features
    m_param.coef0 = 0;
    m_param.nu = 0.5;
    m_param.cache_size = 100;
    m_param.C = 10;//10000;//500;
    m_param.eps = 1e-2;
    m_param.p = 0.1;
    m_param.shrinking = 0;
    m_param.probability = 0;
    m_param.nr_weight = 0;
    m_param.weight_label = NULL;
    m_param.weight = NULL;
    
    m_originalMap = PointCloudType::Ptr(new PointCloudType);
    
    m_mapGridSize = cv::Size(300, 300);
    
    m_minDistCarObstacle = 0.0000001;
    
    m_carWidth = 1.0;
//     m_minDistCarObstacle = 0.1;
}

bool VoronoiSVMPathPlanning::findShortestPath(const PointType& start, const double & startOrientation,
                                                   const PointType & goal, const double & goalOrientation,
                                                   const PointCloudType::Ptr & rtObstacles, bool visualize) {
                                      
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
    
    PointCloudType::Ptr currentMapVoronoi(new PointCloudType);
    pcl::copyPointCloud(*rtObstacles, *currentMapVoronoi);
    if (! findShortestPathVoronoi(start, startOrientation, goal, goalOrientation, currentMapVoronoi, false)) {
        cerr << "Unable to find initial path using Voronoi diagrams" << endl;
        return false;
    }
    
    vector <PointCloudType::Ptr> sites;
    getSites(sites);
    
    PointCloudType::Ptr pathNodes;
    getPathGraph(sites, pathNodes, footprintStart, footprintGoal, rtObstacles);
    
    vector<int> idxMap, idxRT;
    vector<float> distMap, distRT;
    Node startNode, goalNode;
    double cost;
    
    // Creating the KdTree object for the search method of the extraction
    pcl::KdTreeFLANN<PointType>::Ptr treeMap (new pcl::KdTreeFLANN<PointType>);
    treeMap->setInputCloud (pathNodes);
    treeMap->setSortedResults(true);
    
    treeMap->radiusSearch(start, 8 * m_carWidth, idxMap, distMap);
    if (idxMap.size() < 1) {
        cerr << "Unable to find a starting node..." << endl;
        
        return false;
    }
    startNode = m_nodeList[idxMap[0]];

    treeMap->radiusSearch(goal, 8 * m_carWidth, idxMap, distMap);
    if (idxMap.size() < 1) {
        cerr << "Unable to find a goal node..." << endl;
        
        return false;
    }
    goalNode = m_nodeList[idxMap[0]];   
    
//     visualizer(rtObstacles, pathNodes, sites);
    
    if (! lemon::connected(m_graph)) {
        Graph::NodeMap<uint32_t> compMap(m_graph);
        
        lemon::connectedComponents(m_graph, compMap);
        
        if (compMap[startNode] != compMap[goalNode]) {
            cerr << "Unable to find a path: It is impossible to go from " << start << 
                    " to " << goal << " with this configuration." << endl;

            return false; 
        }
    }
    
    lemon::Dijkstra<Graph, EdgeMap> dijkstra(m_graph, *m_distMap);
    dijkstra.init();
    dijkstra.addSource(goalNode);
    dijkstra.start();
    
    cout << "Path found!!!" << endl;
    
    cost = dijkstra.dist(startNode);
    
    m_voronoiPath = PointCloudType::Ptr (new PointCloudType);
    pcl::copyPointCloud(*m_path, *m_voronoiPath);
    m_path->clear();
    
    Node currentNode = startNode;
    m_path->push_back(start);
    while (currentNode != goalNode) {
        currentNode = dijkstra.predNode(currentNode);
        m_path->push_back((*m_nodeMap)[currentNode]);
    }
    m_path->push_back(goal);
    
    cout << "Path length = " << m_path->size() << endl;
    cout << "Path cost = " << cost << endl;
    
    visualizer(rtObstacles, pathNodes, sites);
    
    return true;
    
}

void VoronoiSVMPathPlanning::getPathGraph(vector <PointCloudType::Ptr> & sites, PointCloudType::Ptr & pathNodes,
                                          const vector<PointCloudType::Ptr> & footprintStart,
                                          const vector<PointCloudType::Ptr> & footprintGoal,
                                          const PointCloudType::Ptr & rtObstacles) {
    *(sites[0]) += *(footprintStart[0]);
    *(sites[0]) += *(footprintGoal[0]);
    *(sites[1]) += *(footprintStart[1]);
    *(sites[1]) += *(footprintGoal[1]);
    
    for (Graph::EdgeIt it(m_graph); it != lemon::INVALID; ++it) {
        Edge currentEdge = it;
        m_graph.erase(currentEdge);
    }
    m_nodeList.clear();
    m_distMap = boost::shared_ptr<EdgeMap>(new EdgeMap(m_graph));
    m_nodeMap = boost::shared_ptr<NodeMap>(new NodeMap(m_graph));
    
    CornerLimitsType minCorner = make_double2(DBL_MAX, DBL_MAX), maxCorner = make_double2(DBL_MIN, DBL_MIN), interval;
    
    for (uint32_t i = 0; i < sites.size(); i++) {
        for (PointCloudType::iterator it = sites[i]->begin(); it != sites[i]->end(); it++) {
            if (it->x < minCorner.x) minCorner.x = it->x;
            if (it->y < minCorner.y) minCorner.y = it->y;
            if (it->x > maxCorner.x) maxCorner.x = it->x;
            if (it->y > maxCorner.y) maxCorner.y = it->y;
        }
    }
    
    interval = make_double2(maxCorner.x - minCorner.x, maxCorner.y - minCorner.y);
    
    minCorner.x -= 0.1 * interval.x;
    minCorner.y -= 0.1 * interval.y;
    maxCorner.x += 0.1 * interval.x;
    maxCorner.y += 0.1 * interval.y;
    
    interval = make_double2(maxCorner.x - minCorner.x, maxCorner.y - minCorner.y);
    
    m_distBetweenSamples = 1.5 * sqrt((interval.x / m_mapGridSize.width) * (interval.x / m_mapGridSize.width) +
                                      (interval.y / m_mapGridSize.height) * (interval.y / m_mapGridSize.height));
    
    pathNodes = PointCloudType::Ptr(new PointCloudType);
        
    getBorderFromPointClouds (sites[0], sites[1], minCorner, maxCorner, interval, m_mapGridSize, 0, pathNodes, m_nodeList);
    
    generateRNG(pathNodes, m_nodeList, rtObstacles, false, true);
}

void VoronoiSVMPathPlanning::getSites(vector <PointCloudType::Ptr> & sites)
{
    typedef VD::Locate_result Locate_result;
    typedef VD::Vertex_handle Vertex_handle;
    typedef VD::Face_handle Face_handle;
    typedef VD::Halfedge_handle Halfedge_handle;
    typedef VD::Ccb_halfedge_circulator   Ccb_halfedge_circulator;
    typedef VD::Halfedge_around_vertex_circulator halfedge_around_vertex_circulator;
    
    sites.resize(2);
    sites[0] = PointCloudType::Ptr(new PointCloudType);
    sites[1] = PointCloudType::Ptr(new PointCloudType);
    
    for (PointCloudType::iterator it = m_path->begin() + 1; it != m_path->end(); it++) {

        // We look for the incident edges for the current point in the voronoi diagram
        halfedge_around_vertex_circulator edgeCirculatorStart;
        Locate_result lr = m_vd.locate(Point_2(it->x, it->y));
        if ( Vertex_handle* v = boost::get<Vertex_handle>(&lr) ) {

            edgeCirculatorStart = (*v)->incident_halfedges();
            
        } else if ( Halfedge_handle* e = boost::get<Halfedge_handle>(&lr) ) {
            
            if (((*e)->source()->point().x() == it->x) && ((*e)->source()->point().y() == it->y))
                edgeCirculatorStart = (*e)->source()->incident_halfedges();
            else
                edgeCirculatorStart = (*e)->target()->incident_halfedges();
            
        } else if (Face_handle * f = boost::get<Face_handle>(&lr) ) {

            Ccb_halfedge_circulator ec_start = (*f)->outer_ccb();
            Ccb_halfedge_circulator ec = ec_start;

            const PointType & currentPoint = *it;
            
            do {
                if ( ec->has_target() )  {
                    const PointType pointInFace(ec->target()->point().x(), ec->target()->point().y(), 0.0);
                    
                    if ((currentPoint.x == pointInFace.x) && (currentPoint.y == pointInFace.y)) {
                        
                        edgeCirculatorStart = ec->target()->incident_halfedges();

                        break;
                    }
                }
            } while ( ++ec != ec_start );
        }
        
        // We look for the right faces using the incident edges
        // In other words, from all the incident edges, we look for that originated in the previous point
        Face_handle f1, f2;
        halfedge_around_vertex_circulator edgeCirculator = edgeCirculatorStart;
        
        bool found = false;
        const PointType prevPoint((it - 1)->x, (it - 1)->y, 0.0);
        do {
            const PointType srcPoint(edgeCirculator->source()->point().x(), edgeCirculator->source()->point().y(), 0.0);
            if ((prevPoint.x == srcPoint.x) && (prevPoint.y == srcPoint.y)) {
                
                f1 = edgeCirculator->face();
                f2 = edgeCirculator->opposite()->face();       
                
                found = true;
                
                break;
            }
        } while ( ++edgeCirculator != edgeCirculatorStart );
        
        if (! found) continue;
        
        // We look for the face centroids
        PointType p1(0,0,0), p2(0,0,0);
        double count = 0;
                
        Ccb_halfedge_circulator ec_start = f1->outer_ccb();
        Ccb_halfedge_circulator ec = ec_start;
        do {
            if (ec->has_target()) {
                p1.x += ec->target()->point().x();
                p1.y += ec->target()->point().y();
                count++;
            }
        } while ( ++ec != ec_start );
        p1.x /= count;
        p1.y /= count;
        
        count = 0.0;
        
        ec_start = f2->outer_ccb();
        ec = ec_start;
        do {
            if (ec->has_target()) {
                p2.x += ec->target()->point().x();
                p2.y += ec->target()->point().y();
                count++;
            }
        } while ( ++ec != ec_start );
        p2.x /= count;
        p2.y /= count;
        
        sites[1]->push_back(p1);
        sites[0]->push_back(p2);
    }
}

void VoronoiSVMPathPlanning::visualizer(const PointCloudType::Ptr& currentMap, const PointCloudType::Ptr& pathNodes,
                                        const vector <PointCloudType::Ptr> & sites)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->initCameraParameters();
    viewer->addCoordinateSystem();
    
    PointCloudTypeExt::Ptr currentMapRGB(new PointCloudTypeExt);
    currentMapRGB->reserve(currentMap->size());
    for (PointCloudType::iterator it = currentMap->begin(); it != currentMap->end(); it++) {
        PointTypeExt point;
        
        point.x = it->x;
        point.y = it->y;
        point.z = -1.0;
        
        point.r = 255;
        point.g = 255;
        point.b = 255;
        
        currentMapRGB->push_back(point);
    }

    PointCloudTypeExt::Ptr positiveSites(new PointCloudTypeExt);
    positiveSites->reserve(sites[0]->size());
    for (PointCloudType::iterator it = sites[0]->begin(); it != sites[0]->end(); it++) {
        PointTypeExt point;
        
        point.x = it->x;
        point.y = it->y;
        point.z = 0.0;
        
        point.r = 0;
        point.g = 255;
        point.b = 0;
        
        positiveSites->push_back(point);
    }
    
    PointCloudTypeExt::Ptr negativeSites(new PointCloudTypeExt);
    negativeSites->reserve(sites[1]->size());
    for (PointCloudType::iterator it = sites[1]->begin(); it != sites[1]->end(); it++) {
        PointTypeExt point;
        
        point.x = it->x;
        point.y = it->y;
        point.z = 0.0;
        
        point.r = 255;
        point.g = 0;
        point.b = 255;
        
        negativeSites->push_back(point);
    }
    
    PointCloudTypeExt::Ptr trajectory(new PointCloudTypeExt);
    trajectory->reserve(pathNodes->size());
    
    for (uint32_t i = 0; i < pathNodes->size(); i++) {
        PointTypeExt point;
        
        point.x = pathNodes->at(i).x;
        point.y = pathNodes->at(i).y;
        point.z = 0.0; //-1.0;
        point.r = 0;
        point.g = 255;
        point.b = 0;
        
        trajectory->push_back(point);
    }
    
    PointCloudTypeExt::Ptr pathPointCloud(new PointCloudTypeExt);
    if (m_path.get() != NULL) {
        PointType lastPoint;
        for (uint32_t i = 0; i < m_path->size(); i++) {
            const PointType & point = m_path->at(i);
            
            if (i != 0)
                addLineToPointCloud(lastPoint, point, 0, 255, 255, pathPointCloud, 1.0);
            lastPoint = point;
        }
    }
    
    PointCloudTypeExt::Ptr voronoiPathPointCloud(new PointCloudTypeExt);
    if (m_voronoiPath.get() != NULL) {
        PointType lastPoint;
        for (uint32_t i = 0; i < m_voronoiPath->size(); i++) {
            const PointType & point = m_voronoiPath->at(i);
            
            if (i != 0)
                addLineToPointCloud(lastPoint, point, 255, 255, 0, voronoiPathPointCloud, 1.0);
            lastPoint = point;
        }
    }
    
    pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbNegativeSites(negativeSites);
    viewer->addPointCloud<PointTypeExt> (negativeSites, rgbNegativeSites, "negativeSites");
    
    pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbPositiveSites(positiveSites);
    viewer->addPointCloud<PointTypeExt> (positiveSites, rgbPositiveSites, "positiveSites");
    
    pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbPathCloud(pathPointCloud);
    viewer->addPointCloud<PointTypeExt> (pathPointCloud, rgbPathCloud, "path");
    
    pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbVoronoiPathCloud(voronoiPathPointCloud);
    viewer->addPointCloud<PointTypeExt> (voronoiPathPointCloud, rgbVoronoiPathCloud, "voronoiPath");
    
    pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbCurrentMap(currentMapRGB);
    viewer->addPointCloud<PointTypeExt> (currentMapRGB, rgbCurrentMap, "currentMap");
    
    pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbTrajectory(trajectory);
    viewer->addPointCloud<PointTypeExt> (trajectory, rgbTrajectory, "trajectory");
    
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