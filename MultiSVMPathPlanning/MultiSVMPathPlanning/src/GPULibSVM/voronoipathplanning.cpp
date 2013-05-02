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

#include "voronoipathplanning.h"

#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <vector_functions.h>

#include <lemon/dijkstra.h>
#include <lemon/connectivity.h>

using namespace svmpp;

VoronoiPathPlanning::VoronoiPathPlanning() : SVMPathPlanning()
{
    m_originalMap = PointCloudType::Ptr(new PointCloudType);
    
    m_mapGridSize = cv::Size(300, 300);
}

bool VoronoiPathPlanning::findShortestPathVoronoi(const PointType& start, const double& startOrientation, const PointType& goal, 
                                                  const double& goalOrientation, PointCloudType::Ptr rtObstacles, bool visualize)
{

    *rtObstacles += *m_originalMap;
    
    PointCloudType::Ptr footprintStart, footprintGoal;
    bool startCheck = getFootPrintVoronoi(start, startOrientation, rtObstacles, footprintStart, true);
    bool goalCheck = getFootPrintVoronoi(goal, goalOrientation, rtObstacles, footprintGoal, false);
    
    if (! startCheck) {
        cerr << "Failed to find a path: Current position is too near to an obstacle or colliding with it" << endl;
        
        return false;
    }
    if (! startCheck) {
        cerr << "Failed to find a path: Goal position is not clear" << endl;
        
        return false;
    }
    
    *rtObstacles += *footprintStart;
    *rtObstacles += *footprintGoal;

    bool reached = getVoronoiPath(start, startOrientation, goal, goalOrientation, rtObstacles);
    
    
    if (visualize)
        visualizer(m_pathNodes, rtObstacles);
        
    return reached;
}

void VoronoiPathPlanning::visualizer(const PointCloudType::Ptr& pathNodes, const PointCloudType::Ptr& currentMap)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->initCameraParameters();
    viewer->addCoordinateSystem();
    
    PointCloudTypeExt::Ptr obstaclesCloudRGB(new PointCloudTypeExt);
    obstaclesCloudRGB->reserve(pathNodes->size());
    for (PointCloudType::iterator it = pathNodes->begin(); it != pathNodes->end(); it++) {
        PointTypeExt point;
        
        point.x = it->x;
        point.y = it->y;
        point.z = -1.0;
        
        point.r = 0;
        point.g = 255;
        point.b = 0;
        
        obstaclesCloudRGB->push_back(point);
    }
    
    PointCloudTypeExt::Ptr currentMapRGB(new PointCloudTypeExt);
    currentMapRGB->reserve(currentMap->size());
    for (PointCloudType::iterator it = currentMap->begin(); it != currentMap->end(); it++) {
        PointTypeExt point;
        
        point.x = it->x;
        point.y = it->y;
        point.z = 2.0;
        
        point.r = 255;
        point.g = 0;
        point.b = 255;
        
        currentMapRGB->push_back(point);
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

    pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbObstaclesCloud(obstaclesCloudRGB);
    viewer->addPointCloud<PointTypeExt> (obstaclesCloudRGB, rgbObstaclesCloud, "rtObstacles");
    
    pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbPathCloud(pathPointCloud);
    viewer->addPointCloud<PointTypeExt> (pathPointCloud, rgbPathCloud, "path");
    
    pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbCurrentMap(currentMapRGB);
    viewer->addPointCloud<PointTypeExt> (currentMapRGB, rgbCurrentMap, "currentMap");
        
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

bool VoronoiPathPlanning::getVoronoiPath(const PointType& start, const double& startOrientation, const PointType& goal, 
                                         const double& goalOrientation, const PointCloudType::Ptr & currentMap)
{
    getVoronoiDiagram(currentMap);
    
    Node startNode, goalNode;
    double startMinDist = DBL_MAX, goalMinDist = DBL_MAX;
    for (vector<Node>::iterator it = m_nodeList.begin(); it != m_nodeList.end(); it++) {
        const PointType & currentPoint = (*m_nodeMap)[*it];
        const double & startDist = pcl::euclideanDistance(currentPoint, start);
        const double & goalDist = pcl::euclideanDistance(currentPoint, goal);
        if (startDist < startMinDist) {
            startNode = *it;
            startMinDist = startDist;
        }
        
        if (goalDist < goalMinDist) {
            goalNode = *it;
            goalMinDist = goalDist;
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
    m_path->clear();
    m_path->push_back(start);
    while (currentNode != goalNode) {
        currentNode = dijkstra.predNode(currentNode);
        m_path->push_back((*m_nodeMap)[currentNode]);
    }
    m_path->push_back(goal);
        
    return true;
}

void VoronoiPathPlanning::getVoronoiDiagram(const PointCloudType::Ptr& currentMap)
{
    vector< pair<uint32_t, uint32_t> > edges;
    
    // Nodes are inserted into Voronoi diagram
    for (uint32_t i = 0; i < currentMap->size(); i++) {
        const Site_2 site(currentMap->at(i).x, currentMap->at(i).y);
        m_vd.insert(site);
    }

    // The list of vertices generated by Voronoi diagram are pushed into the graph
    PointCloudType::Ptr pathNodes(new PointCloudType);
    pathNodes->reserve(m_vd.number_of_vertices());
    m_nodeList.clear();
    m_nodeList.reserve(m_vd.number_of_vertices());
    for (VtxIterator it = m_vd.vertices_begin(); it!= m_vd.vertices_end(); it++) {
        PointType point(it->point().x(), it->point().y(), 0.0);
        
        pathNodes->push_back(point);
        
        Node node = m_graph.addNode();
        (*m_nodeMap)[node] = point;
        m_nodeList.push_back(node);
    }
    
    vector<int> pointIdxNKNSearch;
    vector<float> pointNKNSquaredDistance;
    pcl::KdTreeFLANN<PointType>::Ptr tree (new pcl::KdTreeFLANN<PointType>);
    tree->setInputCloud(pathNodes);
    tree->setSortedResults(true);
    
    for (EdgeIterator it = m_vd.edges_begin(); it != m_vd.edges_end(); it++) {
        if (it->has_source() && it->has_target()) {
            const PointType pointA(it->source()->point().x(), it->source()->point().y(), 0.0);
            const PointType pointB(it->target()->point().x(), it->target()->point().y(), 0.0);
            
            int idx1, idx2;
            tree->nearestKSearch(pointA, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
            idx1 = pointIdxNKNSearch[0];
            tree->nearestKSearch(pointB, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
            idx2 = pointIdxNKNSearch[0];
            
            edges.push_back(make_pair<uint32_t, uint32_t>(idx1, idx2));
        }
    }
    
    checkSegments(pathNodes, m_nodeList, currentMap, edges, true);
}

bool VoronoiPathPlanning::getFootPrintVoronoi(const PointType& position, const double& orientation, 
                                              const PointCloudType::Ptr& rtObstacles, 
                                              PointCloudType::Ptr & footprint,
                                              const bool & isStart)
{
    vector< PointCloudType::Ptr > tmpFootprint;
    if (!getFootPrint(position, orientation, rtObstacles, tmpFootprint))
        return false;
    
    footprint = PointCloudType::Ptr(new PointCloudType);
    PointType p1, p2;
    if (isStart) {
        p1 = tmpFootprint[0]->at(0);
        p2 = tmpFootprint[1]->at(0);
    } else {
        p1 = tmpFootprint[0]->at(tmpFootprint[0]->size() - 1);
        p2 = tmpFootprint[1]->at(tmpFootprint[1]->size() - 1);        
    }
        
    double dist = pcl::euclideanDistance(p1, p2);
    
    for (double i = 0; i <= m_carWidth; i += m_minPointDistance / 2.0) {
        const PointType p(p1.x + i * (p2.x - p1.x),
                            p1.y + i * (p2.y - p1.y),
                            0.0);
        
        footprint->push_back(p);
    }
    *footprint += *tmpFootprint[0];
    *footprint += *tmpFootprint[1];
    
    return true;
}
