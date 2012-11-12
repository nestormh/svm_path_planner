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

#include "ccpdtracking.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

CCPDTracking::CCPDTracking(const t_SGBM_params & stereoParams, boost::shared_ptr<pcl::visualization::PCLVisualizer> & viewer, 
                           const bool & useDispImg, const bool & output, const bool & loadMatchesFromFile) : 
                            m_viewer(viewer), m_stereoParams(stereoParams), m_useDispImg(useDispImg), m_output(output), 
                            m_showCloud(false), m_loadMatchesFromFile(loadMatchesFromFile) {
    
    m_stereoParams = stereoParams;
    m_output = output;        
}

CCPDTracking::CCPDTracking(const CCPDTracking& other) {
}

CCPDTracking::~CCPDTracking() {
}

void CCPDTracking::update(cv::Mat left, cv::Mat right, const double & groundThresh, const double & bgThresh, const t_Camera_params & paramsLeft, const t_Camera_params & paramsRight) {
    
    m_groundThresh = groundThresh;
    m_bgThresh = bgThresh;
    m_paramsLeft = paramsLeft;
    m_paramsRight = paramsRight;
    
    cv::Mat leftGray(left.rows, left.cols, CV_8UC1);    
    cv::Mat rightGray(left.rows, left.cols, CV_8UC1);    
    cv::Mat disp8(left.rows, left.cols, CV_64F);
    cvtColor(left, leftGray, CV_BGR2GRAY);
    cvtColor(right, rightGray, CV_BGR2GRAY);
    
    
    cv::StereoSGBM stereo(m_stereoParams.minDisparity, m_stereoParams.numDisparities, m_stereoParams.SADWindowSize,
                          m_stereoParams.SADWindowSize * m_stereoParams.SADWindowSize * m_stereoParams.P1, 
                          m_stereoParams.SADWindowSize * m_stereoParams.SADWindowSize * m_stereoParams.P2, 
                          m_stereoParams.disp12MaxDiff, m_stereoParams.preFilterCap, m_stereoParams.uniquenessRatio,
                          m_stereoParams.speckleWindowSize, m_stereoParams.speckleRange, m_stereoParams.fullDP);    
    
    cv::Mat disp;
    stereo(leftGray, rightGray, disp);  
    
    disp.convertTo(disp8, CV_64F, 1./16.);
    
    //Create point cloud and fill it    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    Eigen::MatrixXd mPointIn(4,1);
    Eigen::MatrixXd mPointOut;
    
    Eigen::MatrixXd mRt = Camera2World();
    std::cout << "mRt = " << std::endl << mRt << std::endl;
    
    for (int i = 0; i < left.rows; i++) {
        uchar* rgb_ptr = left.ptr<uchar>(i);
        double* disp_ptr = disp8.ptr<double>(i);
        
        for (int j = 0; j < left.cols; j++) {
            
            double d = disp_ptr[j];
            
            if (d <= 0) continue;
            
            double norm = (double)d / (double)(m_paramsLeft.ku * (m_paramsLeft.y - m_paramsRight.y));
            
            //Get 3D coordinates
            pcl::PointXYZRGB point;
            
            if (! m_useDispImg) {
                mPointIn << 1.0 / norm
                           ,(((m_paramsLeft.u0 - j) / m_paramsLeft.ku) / norm)
                           ,(((m_paramsLeft.v0 - i) / m_paramsLeft.kv) / norm)
                           ,1.0;
            } else {
                mPointIn << j, i, d, 1.0;
            }
            
            mPointOut = mRt * mPointIn;
            
            point.x = mPointOut.data()[0]/* + m_paramsLeft.x*/;
            point.y = mPointOut.data()[1]/* + m_paramsLeft.y*/;
            point.z = mPointOut.data()[2]/* + m_paramsLeft.z*/;
            
            point.b = rgb_ptr[3*j];
            point.g = rgb_ptr[3*j + 1];
            point.r = rgb_ptr[3*j + 2];
            
//             if ((point.z > 0.2) && (point.x < 20))
//             if (point.z > 0.2)
            point_cloud_ptr->points.push_back(point);
                        
        }
    }
    point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
    point_cloud_ptr->height = 1; 
    
    removeGround(/*point_cloud_ptr*/);
    
    downsample(point_cloud_ptr);
    
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud_ptr);
    if (! m_viewer->updatePointCloud<pcl::PointXYZRGB> (point_cloud_ptr, rgb, "point_cloud_ptr")) {
        m_viewer->addPointCloud<pcl::PointXYZRGB> (point_cloud_ptr, rgb, "point_cloud_ptr");
    } 
    
//     update(point_cloud_ptr);
    
}

void CCPDTracking::update(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & newPointCloud) {
    
    m_pOldPointCloud = m_pNewPointCloud;
    m_pNewPointCloud = newPointCloud;
    
    if (m_pOldPointCloud == NULL) {
        std::cout << "No hay nubes de puntos anteriores" << std::endl;
        return;
    }
    
    if (! callMatlabCPD()) {
        std::cerr << "Frame rechazado" << std::endl;
        m_pNewPointCloud = m_pOldPointCloud; // Así evitamos que en la próxima iteración empleemos la nube del frame rechazado
    }
        
    // TODO: Si el porcentaje de puntos emparejados es demasiado pequeño, rechaza el frame
    
    
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(m_pNewPointCloud);
    if (! m_viewer->updatePointCloud<pcl::PointXYZRGB> (m_pNewPointCloud, rgb, "newPointCloud")) {
        m_viewer->addPointCloud<pcl::PointXYZRGB> (m_pNewPointCloud, rgb, "newPointCloud");
    } 
       
}

bool CCPDTracking::callMatlabCPD() {
    std::string resultsFileName;
    if (m_loadMatchesFromFile) {
        std::stringstream ss;
        ss << m_CPD_opts.basePath << "/" << m_CPD_opts.sequenceName << "/corresp/corresp" << m_CPD_opts.iteration << ".txt";
        resultsFileName = ss.str();        
    } else {
        resultsFileName = "/tmp/CPD/outputCPD.txt";
    }
    
    std::ifstream resultsFile(resultsFileName.c_str());
    
    if (!m_loadMatchesFromFile || ! resultsFile.good()) {        
        
        // TODO: Eliminar esto, sólo depuración        
//         m_pNewPointCloud->erase(m_pNewPointCloud->begin() + 100, m_pNewPointCloud->end());
//         m_pOldPointCloud->erase(m_pOldPointCloud->begin() + 100, m_pOldPointCloud->end());
        // TODO: Fin del aviso
        
        std::ofstream fout("/tmp/CPD/paramsCPD.txt", ios::out | ios::trunc);
        fout << "method = " << m_CPD_opts.method << std::endl;
        fout << "corresp = " << m_CPD_opts.estimateCorresp << std::endl;
        fout << "normalize = " << m_CPD_opts.normalize << std::endl;
        fout << "max_it = " << m_CPD_opts.max_it << std::endl;
        fout << "tol = " << m_CPD_opts.tolerance << std::endl;
        fout << "viz = " << m_CPD_opts.visualize << std::endl;
        fout << "outliers = " << m_CPD_opts.outliersWeight << std::endl;
        fout << "fgt = " << m_CPD_opts.fgt << std::endl;
        fout << "rot = " << m_CPD_opts.justRotation << std::endl;
        fout << "scale = " << m_CPD_opts.estimateScaling << std::endl;
        fout << "beta = " << m_CPD_opts.beta << std::endl;
        fout << "lambda = " << m_CPD_opts.lambda << std::endl;
        fout << "nDims = " << m_CPD_opts.nDims << std::endl;
        fout << "distThresh = " << m_CPD_opts.distThresh << std::endl;
        fout << "saveOutput = " << m_CPD_opts.saveOutput << std::endl;
        fout << "basePath = " << m_CPD_opts.basePath << std::endl;
        fout << "sequenceName = " << m_CPD_opts.sequenceName << std::endl;
        fout << "iteration = " << m_CPD_opts.iteration << std::endl;
        
        fout << "m_pOldPointCloudSize = " << m_pOldPointCloud->size() << std::endl;
        fout << "m_pNewPointCloudSize = " << m_pNewPointCloud->size() << std::endl;
        fout.close();
        
        fout.open("/tmp/CPD/dataCPD.txt", ios::out | ios::trunc);
        for (uint32_t i = 0; i < m_pOldPointCloud->size(); i++) {
            fout << m_pOldPointCloud->at(i).x << "  " << m_pOldPointCloud->at(i).y << " " << m_pOldPointCloud->at(i).z << " " <<
                    (uint32_t)m_pOldPointCloud->at(i).r << "    " << (uint32_t)m_pOldPointCloud->at(i).g << "   " << (uint32_t)m_pOldPointCloud->at(i).b << std::endl;
        }

        for (uint32_t i = 0; i < m_pNewPointCloud->size(); i++) {
            fout << m_pNewPointCloud->at(i).x << "  " << m_pNewPointCloud->at(i).y << " " << m_pNewPointCloud->at(i).z << " " <<
                    (uint32_t)m_pNewPointCloud->at(i).r << "    " << (uint32_t)m_pNewPointCloud->at(i).g << "   " << (uint32_t)m_pNewPointCloud->at(i).b << std::endl;
        }

        fout.close();
        
        system("/home/nestor/gold/apps/cpd/matlab/CPDexec/run_CPDexec.sh /opt/MATLAB/MATLAB_Compiler_Runtime/v715/");
    }
    resultsFile.close();
    
    resultsFile.open(resultsFileName.c_str());
    
    if (! resultsFile.good()) {
        std::cerr << "Se produjo un error al intentar abrir el fichero " << resultsFileName << ", en la iteración " << m_CPD_opts.iteration << std::endl;
        return false;
    }
    
    // Nos saltamos la cabecera
    for (uint32_t i = 0; i < 15; i++) {
        resultsFile.ignore(1024, '\n');
    }
    
    std::vector<uint32_t> correspondences;
    correspondences.reserve(m_pOldPointCloud->size());
    int32_t val;
    while (resultsFile.good()) {
        resultsFile >> val;
        correspondences.push_back(val);        
    }
    std::cout << "Finalizo" << std::endl;
    
    resultsFile.close();
    
    return true;
}

inline Eigen::MatrixXd CCPDTracking::Camera2World() {
    Eigen::MatrixXd rotoTranslationMatrix(3,4);
    
    t_Camera_params params = m_paramsLeft;
    
    rotoTranslationMatrix << cos(params.pitch)*cos(params.yaw)
                           , sin(params.roll)*sin(params.pitch)*cos(params.yaw)-cos(params.roll)*sin(params.yaw)
                           , cos(params.roll)*sin(params.pitch)*cos(params.yaw)+sin(params.roll)*sin(params.yaw)
                           , params.x

                           , cos(params.pitch)*sin(params.yaw)
                           , sin(params.roll)*sin(params.pitch)*sin(params.yaw)+cos(params.roll)*cos(params.yaw)
                           , cos(params.roll)*sin(params.pitch)*sin(params.yaw)-sin(params.roll)*cos(params.yaw)
                           , params.y
    
                           , -sin(params.pitch)
                           , sin(params.roll)*cos(params.pitch)
                           , cos(params.roll)*cos(params.pitch)
                           , params.z;
                           
    return rotoTranslationMatrix;
}

 inline void CCPDTracking::removeGround() {
    std::cout << "Eliminando suelo" << std::endl;
    
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpPointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
//     
// //     for (uint32_t i = 0; i < /*pointCloud->size()*/500; i += 1) {
// // //         if ((pointCloud->points.at(i).z < m_groundThresh) && (pointCloud->points.at(i).x < m_bgThresh))
// //             pcl::PointXYZRGB p = pointCloud->points.at(i);
// //             p.x *= 1024;
// //             p.y *= 1024;
// //             p.z *= 1024;
// //             tmpPointCloud->push_back(p);
// // //             tmpPointCloud->push_back(pointCloud->points.at(i));
// //     }        
//     tmpPointCloud->width    = 500;
//     tmpPointCloud->height   = 1;
//     tmpPointCloud->is_dense = false;
//     tmpPointCloud->points.resize (tmpPointCloud->width * tmpPointCloud->height);
//     for (size_t i = 0; i < tmpPointCloud->points.size (); ++i) {
//       tmpPointCloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0);
//       tmpPointCloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0);
//       if( i % 2 == 0)
//         tmpPointCloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0);
//       else
//         tmpPointCloud->points[i].z = -1 * (tmpPointCloud->points[i].x + tmpPointCloud->points[i].y);
//   }
//     
//     
//     if (tmpPointCloud->size() < 4) {
//         std::cerr << "Puntos insuficientes para continuar con la eliminación de suelo..." << std::endl;
//         return;
//     }
// 
//     pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr model_plane (new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> (tmpPointCloud));    
//     pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac (model_plane);
//     ransac.setDistanceThreshold (1024);
// //     ransac.setProbability(.99);
// //     ransac.setMaxIterations(500);
//     
//     std::cout << "dist = " << ransac.getDistanceThreshold() << 
//                  ", prob = " << ransac.getProbability() << 
//                  ", maxIt = " << ransac.getMaxIterations() << std::endl;
//     ransac.computeModel();
//     std::vector<int> inliers;
//     ransac.getInliers(inliers);    
// //     pcl::copyPointCloud<pcl::PointXYZRGB>(*tmpPointCloud, inliers, *pointCloud);
// //     
//     std::cout << "Encontrados: " << inliers.size() << std::endl;
// 
//     pointCloud = tmpPointCloud;
//     //     
// //     sample_consensus::RANSAC(ground, worldPoints, 0.01, 0.99, 500);
// //     
// //     std::vector<Point3d> inliers;
// //     inliers.reserve(worldPoints.size());
// //     
// //     for(unsigned int p = 0; p < initialPoints.size(); ++p)
// //         if(std::abs(ground.Distance(initialPoints[p])) < thresh)
// //             inliers.push_back(initialPoints[p]);
// // 
// //     if (inliers.size() == 0) {
// //         filteredPoints = initialPoints;
// //         return;
// //     }
// //         
// //     sample_consensus::estimators::Plane estimator;
// // 
// //     estimator.Optimize(ground, inliers);
// //     
// //     // the detected plane might be upside down
// //     if(ground.Normal().z < 0)
// //         ground = math::Plane3d(-ground.Normal().x, -ground.Normal().y, -ground.Normal().z, -ground.Distance(Point3d(0, 0, 0)));
// //         
// //     filteredPoints = initialPoints;
// //     
// //     for(unsigned int p = 0; p < initialPoints.size(); ++p)
// //         if(ground.Distance(initialPoints[p]) < thresh /*0.1*/)
// //             filteredPoints[p] = inf;   
    
     // initialize PointClouds
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr final (new pcl::PointCloud<pcl::PointXYZ>);

  // populate our PointCloud with points
  cloud->width    = 500;
  cloud->height   = 1;
  cloud->is_dense = false;
  cloud->points.resize (cloud->width * cloud->height);
  for (size_t i = 0; i < cloud->points.size (); ++i)
  {
      cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0);
      cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0);
      if( i % 2 == 0)
        cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0);
      else
        cloud->points[i].z = -1 * (cloud->points[i].x + cloud->points[i].y);
  }

  std::vector<int> inliers;

  // created RandomSampleConsensus object and compute the appropriated model
//   pcl::SampleConsensusModelNormalPlane<pcl::PointXYZ, pcl::Normal>::Ptr
//     model_p (new pcl::SampleConsensusModelNormalPlane<pcl::PointXYZ, pcl::Normal> (cloud));
  pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr
    model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_p);
    ransac.setDistanceThreshold (.01);
    ransac.computeModel();
    ransac.getInliers(inliers);

  // copies all inliers of the model computed to another PointCloud
  pcl::copyPointCloud<pcl::PointXYZ>(*cloud, inliers, *final);
    
    
    std::cout << "Suelo eliminado" << std::endl;
}

inline void CCPDTracking::downsample(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & pointCloud) {    
    std::cout << "Downsampling..." << std::endl;
    
    // TODO: Downsample del número de puntos
    
//     std::cout << "Downsampling..." << std::endl;
//     pcl::VoxelGrid<pcl::PointXYZRGB> ds;  //create downsampling filter 
//     std::cout << "1..." << std::endl;
//     ds.setInputCloud (point_cloud_ptr); 
//     std::cout << "2..." << std::endl;
//     ds.setLeafSize (0.00, 0.00, 0.00); 
//     std::cout << "3..." << std::endl;
//     pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
//     ds.filter (*tmpCloud); 
//     std::cout << point_cloud_ptr->size() << std::endl;
//     std::cout << tmpCloud->size() << std::endl;
//     point_cloud_ptr = tmpCloud; 
//     std::cout << "Downsampling finished..." << std::endl;
}