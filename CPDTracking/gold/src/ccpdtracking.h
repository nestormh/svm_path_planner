/*
    Copyright (c) 2012, Néstor Morales Hernández <nestor@isaatc.ull.es>
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

    THIS SOFTWARE IS PROVIDED BY Néstor Morales Hernández <email> ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Néstor Morales Hernández <email> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * \file ccpdtracking.h
 * \author Néstor Morales Hernández (nestor@isaatc.ull.es)
 * \date 2012-10-11
 */

#ifndef CCPDTRACKING_H
#define CCPDTRACKING_H
#include <vector>
// #include <Data/Math/points.h>

#include <opencv2/opencv.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/cloud_viewer.h>

#include <Eigen/Dense>

typedef struct {        
    int minDisparity;
    int numDisparities;
    int SADWindowSize; 
    int P1; 
    int P2; 
    int disp12MaxDiff; 
    int preFilterCap; 
    int uniquenessRatio;
    
    int speckleWindowSize; 
    int speckleRange; 
    bool fullDP;
} t_SGBM_params;

typedef struct {        
    int width, height;
    double u0, v0;
    double ku, kv;
    double x, y, z;
    double yaw, pitch, roll;
    
    
} t_Camera_params;

typedef struct {
    std::string method; // 'rigid','affine','nonrigid','nonrigid_lowrank']
    bool estimateCorresp;
    bool normalize;
    uint32_t max_it;
    double tolerance;
    bool visualize;
    double outliersWeight;
    uint32_t fgt;
    
    // Rigid registration options
    bool justRotation;
    bool estimateScaling;
    
    // Non-Rigid registration options
    double beta;
    double lambda;
    
    // Other
    uint32_t nDims;
    double distThresh;
    bool saveOutput;
    std::string basePath;
    std::string sequenceName;
    uint32_t iteration;
} t_CPD_opts;

class CCPDTracking
{

public:
    CCPDTracking(const t_SGBM_params & stereoParams, boost::shared_ptr<pcl::visualization::PCLVisualizer> & viewer, 
                 const bool & useDispImg, const bool & output, const bool & loadMatchesFromFile);
    
    CCPDTracking(const CCPDTracking& other);
    virtual ~CCPDTracking();
    
    void update(cv::Mat left, cv::Mat right, const double & groundThresh, const double & bgThresh, const t_Camera_params & paramsLeft, const t_Camera_params & paramsRight);
    void activatePCLVisualization() { m_showCloud = true; };
    
    void setCPDOptions(const t_CPD_opts & CPD_opts) { m_CPD_opts = CPD_opts; };
    
private:    
    void update(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & newPointCloud);    
    Eigen::MatrixXd Camera2World();
    void removeGround();
    void downsample(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & pointCloud);
    
    bool callMatlabCPD();
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_pOldPointCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_pNewPointCloud;
    
    bool m_useDispImg;
    bool m_output;
    bool m_showCloud;
    double m_groundThresh;
    double m_bgThresh;
    
    t_SGBM_params m_stereoParams;
    t_Camera_params m_paramsLeft, m_paramsRight;
    
    bool m_loadMatchesFromFile;
    
    t_CPD_opts m_CPD_opts;
    
    boost::shared_ptr<pcl::visualization::PCLVisualizer> m_viewer;

};

#endif // CCPDTRACKING_H
