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


#ifndef SVMPATHPLANNING_H
#define SVMPATHPLANNING_H

#include "svm.h"

#include <string.h>
#include <fstream>

#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include <opencv2/opencv.hpp>

#define NDIMS 2

using namespace std;

class SVMPathPlanning {
    
    typedef pcl::PointXYZRGB PointType;
    typedef pcl::PointCloud<PointType> PointCloudType;
    typedef pair<double, double> CornerLimitsType;

public:
    SVMPathPlanning();
    SVMPathPlanning ( const SVMPathPlanning& other );
    virtual ~SVMPathPlanning();
    
    void testSingleProblem();
    
private:
    void loadDataFromFile ( const std::string & fileName,
                            PointCloudType::Ptr & X,
                            PointCloudType::Ptr & Y );
    
    void visualizeClasses ( const PointCloudType::Ptr & X, const PointCloudType::Ptr & Y );
    void getBorderFromPointClouds ( const PointCloudType::Ptr & X, const PointCloudType::Ptr & Y,
                                    CornerLimitsType & minCorner, CornerLimitsType & maxCorner,
                                    double resolution);
    
    struct svm_parameter m_param;
    struct svm_problem m_problem;
};
    
    

#endif // SVMPATHPLANNING_H
