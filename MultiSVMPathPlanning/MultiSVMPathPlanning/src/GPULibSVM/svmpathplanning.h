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
// #include "nodexyzrgb.h"

#include <string.h>
#include <fstream>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include <opencv2/opencv.hpp>

#include <vector_types.h>

#define NDIMS 2

using namespace std;

namespace svmpp {

extern "C"
void launchSVMPrediction(const svm_model * &model, 
                         const unsigned int & rows, const unsigned int & cols, 
                         unsigned char * &h_data);

extern "C"
void GPUPredictWrapper(int m, int n, int k, float kernelwidth, const float *Test, 
                       const float *Svs, float * alphas,float *prediction, float beta,
                       float isregression, float * elapsed);

typedef double2 CornerLimitsType;
typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> PointCloudType;
typedef pcl::PointXYZRGB PointTypeExt;
typedef pcl::PointCloud<PointTypeExt> PointCloudTypeExt;
typedef boost::property<boost::edge_weight_t, double> EdgeWeightProperty;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, PointType, EdgeWeightProperty> Graph;
    
class SVMPathPlanning {
    
public:
    SVMPathPlanning();
    SVMPathPlanning ( const SVMPathPlanning& other );
    virtual ~SVMPathPlanning();
    
    void testSingleProblem();
    void obtainGraphFromMap(const PointCloudTypeExt::Ptr & inputCloud);
    
private:
    void loadDataFromFile ( const std::string & fileName,
                            PointCloudType::Ptr & X,
                            PointCloudType::Ptr & Y );
    
    void addLineToPointCloud(const PointType& p1, const PointType& p2, 
                             const uint8_t & r, const uint8_t & g, const uint8_t  & b,
                             PointCloudTypeExt::Ptr &linesPointCloud);
    void visualizeClasses(const std::vector< PointCloudType::Ptr > & classes);
    void getBorderFromPointClouds (PointCloudType::Ptr & X, PointCloudType::Ptr & Y );
    void getContoursFromSVMPrediction(const svm_model * &model, const CornerLimitsType & interval);
    
    void clusterize(const PointCloudTypeExt::Ptr & pointCloud, 
                    std::vector< PointCloudType::Ptr > & classes);
    
    void generateRNG();
  
    struct svm_parameter m_param;
    struct svm_problem m_problem;
    
    double m_minPointDistance;
    cv::Size m_gridSize;
    double m_minDistBetweenObstacles;
    double m_distBetweenSamples;
    
    PointCloudType::Ptr m_existingNodes;
    
    CornerLimitsType m_minCorner;
    CornerLimitsType m_maxCorner;
    
    Graph m_graph;
    
    vector< pair<uint32_t, uint32_t> > m_matches;
};
    
}

#endif // SVMPATHPLANNING_H
