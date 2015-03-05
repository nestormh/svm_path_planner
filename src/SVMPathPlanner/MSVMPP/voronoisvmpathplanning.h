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

#ifndef VORONOISVMPATHPLANNING_H
#define VORONOISVMPATHPLANNING_H

#include "voronoipathplanning.h"

namespace svmpp {
class VoronoiSVMPathPlanning : public VoronoiPathPlanning
{
public:
    VoronoiSVMPathPlanning();
    bool findShortestPath(const PointType& start, const double & startOrientation,
                          const PointType & goal, const double & goalOrientation,
                          const PointCloudType::Ptr & rtObstacles, bool visualize);
    
private:
//     void visualizer(const PointCloudType::Ptr& currentMap, const PointCloudType::Ptr& pathNodes,
//                     const vector <PointCloudType::Ptr> & sites);
    
    void getSites(vector <PointCloudType::Ptr> & sites);
    
    void getPathGraph(vector <PointCloudType::Ptr> & sites, PointCloudType::Ptr & pathNodes,
                      const vector<PointCloudType::Ptr> & footprintStart,
                      const vector<PointCloudType::Ptr> & footprintGoal,
                      const PointCloudType::Ptr & rtObstacles);
    
    PointCloudType::Ptr m_voronoiPath;
};
}
#endif // VORONOISVMPATHPLANNING_H
