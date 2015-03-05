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


#ifndef VORONOIPATHPLANNING_H
#define VORONOIPATHPLANNING_H

// CGAL includes for defining the Voronoi diagram adaptor
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Voronoi_diagram_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_policies_2.h>

#include "svmpathplanning.h"

namespace svmpp {

class VoronoiPathPlanning : public SVMPathPlanning
{
public:
    VoronoiPathPlanning();
    
    bool findShortestPath(const PointType& start, const double& startOrientation, const PointType& goal, 
                            const double& goalOrientation, const PointCloudType::Ptr & rtObstacles, bool visualize);
    
    bool findShortestPathVoronoi(const PointType& start, const double & startOrientation,
                                 const PointType & goal, const double & goalOrientation,
                                 const PointCloudType::Ptr & rtObstacles, bool visualize);
    
protected:
    // typedefs for defining the adaptor
    typedef CGAL::Exact_predicates_inexact_constructions_kernel                  K;
    typedef CGAL::Delaunay_triangulation_2<K>                                    DT;
    typedef CGAL::Delaunay_triangulation_adaptation_traits_2<DT>                 AT;
    typedef CGAL::Delaunay_triangulation_caching_degeneracy_removal_policy_2<DT> AP;
    typedef CGAL::Voronoi_diagram_2<DT,AT,AP>                                    VD;
    
    // typedef for the result type of the point location
    typedef AT::Site_2                    Site_2;
    typedef AT::Point_2                   Point_2;
    
    typedef VD::Vertex_iterator           VtxIterator;
    typedef VD::Edge_iterator             EdgeIterator;
    
    VD m_vd;
    
private:
//     void visualizer(const PointCloudType::Ptr& pathNodes, const PointCloudType::Ptr& currentMap); 
    
    bool getVoronoiPath(const PointType& start, const double& startOrientation, const PointType& goal, 
                        const double& goalOrientation, const PointCloudType::Ptr & currentMap);
    
    void getVoronoiDiagram(const PointCloudType::Ptr & currentMap, PointCloudType::Ptr & pathNodes);
    
    bool getFootPrintVoronoi(const PointType & position, const double & orientation, 
                             const PointCloudType::Ptr & rtObstacles, PointCloudType::Ptr & footprint,
                             const bool & isStart);
    
};

}
#endif // VORONOIPATHPLANNING_H
