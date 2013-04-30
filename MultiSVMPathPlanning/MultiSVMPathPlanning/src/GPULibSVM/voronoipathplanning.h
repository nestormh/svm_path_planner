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


#ifndef VORONOIPATHPLANNING_H
#define VORONOIPATHPLANNING_H

#include "svmpathplanning.h"

namespace svmpp {

class VoronoiPathPlanning : public SVMPathPlanning
{
public:
    VoronoiPathPlanning();
    
    bool findShortestPathVoronoi(const PointType& start, const double & startOrientation,
                                const PointType & goal, const double & goalOrientation,
                                PointCloudType::Ptr rtObstacles, bool visualize);
private:
    void visualizer(const PointCloudType::Ptr& pathNodes, const PointCloudType::Ptr& currentMap); 
    
    bool getVoronoiPath(const PointType& start, const double& startOrientation, const PointType& goal, 
                        const double& goalOrientation, const PointCloudType::Ptr & currentMap);
    
    void getVoronoiDiagram(const PointCloudType::Ptr & currentMap);
    
    bool getFootPrintVoronoi(const PointType & position, const double & orientation, 
                             const PointCloudType::Ptr & rtObstacles, PointCloudType::Ptr & footprint,
                             const bool & isStart);
    
};

}
#endif // VORONOIPATHPLANNING_H
