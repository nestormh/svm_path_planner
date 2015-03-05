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


#ifndef SVMPATHPLANNINGSINGLE_H
#define SVMPATHPLANNINGSINGLE_H

#include "svmpathplanning.h"

namespace svmpp {
    
class SVMPathPlanningSingle : public SVMPathPlanning {
public:
    SVMPathPlanningSingle();
    
    void setMap(const PointCloudType::Ptr & inputCloud);
    bool findShortestPath(const PointType& start, const double & startOrientation,
                          const PointType & goal, const double & goalOrientation,
                          const PointCloudType::Ptr & rtObstacles, bool visualize);
private:
    bool isLeftFromLine(const PointType & line1, const PointType & line2, const PointType & point);
    
//     void visualizer(const PointCloudType::Ptr & class1, const PointCloudType::Ptr & class2, 
//                     const PointType & start, const PointType & goal);
    
    bool getNewPattern(vector < vector <bool> > & visitedPatterns, const vector<bool> & fixedClasses);
    
    bool getPath(const vector <bool> & currentPattern,
                 const vector< PointCloudType::Ptr > & classes,
                 const vector<PointCloudType::Ptr> & footprintStart, 
                 const vector<PointCloudType::Ptr> & footprintGoal,
                 const PointType & p1, const PointType & p2,
                 PointCloudType::Ptr & path, double & cost,
                 const bool & visualize);
    
    bool isFixedClass(const PointCloudType::Ptr & pointCloud, const double & threshold, const double & isLower);
    
    void getAvailablePatterns(const vector<bool> & currentPattern, const vector<bool> & fixedClasses, 
                              vector <vector <bool> > & availablePatterns, const uint32_t & idx);
    
    uint32_t m_Np;
    double m_Dv;
    uint32_t m_maxIterations;
};

}

#endif // SVMPATHPLANNINGMIURA_H
