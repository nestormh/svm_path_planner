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


#include "svmpathplanning.h"

#include <time.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

SVMPathPlanning::SVMPathPlanning()
{
    // default values
    m_param.svm_type = C_SVC;
    m_param.kernel_type = RBF;
    m_param.degree = 3;
    m_param.gamma = 300;        // 1/num_features
    m_param.coef0 = 0;
    m_param.nu = 0.5;
    m_param.cache_size = 100;
    m_param.C = 500;
    m_param.eps = 1e-3;
    m_param.p = 0.1;
    m_param.shrinking = 0;
    m_param.probability = 0;
    m_param.nr_weight = 0;
    m_param.weight_label = NULL;
    m_param.weight = NULL;
//     cross_validation = 0;
    
    m_minPointDistance = 0.005;
    m_resolution = 400;

}

SVMPathPlanning::SVMPathPlanning ( const SVMPathPlanning& other )
{

}

SVMPathPlanning::~SVMPathPlanning()
{

}

void SVMPathPlanning::testSingleProblem()
{
    PointCloudType::Ptr X, Y;
    
    loadDataFromFile ( "/home/nestor/Dropbox/projects/MultiSVMPathPlanning/out1.txt", X, Y );
    
    CornerLimitsType minCorner = make_pair<double, double> ( 0.0, 0.0 );
    CornerLimitsType maxCorner = make_pair<double, double> ( 1.0, 1.0 );
    
    reducePointCloud(X);
    reducePointCloud(Y);
//     visualizeClasses(X, Y);
    
    getBorderFromPointClouds ( X, Y, minCorner, maxCorner, 400.0);
}

void SVMPathPlanning::loadDataFromFile (const std::string & fileName,
                                        PointCloudType::Ptr & X,
                                        PointCloudType::Ptr & Y )
{
    
    X = PointCloudType::Ptr ( new PointCloudType );
    Y = PointCloudType::Ptr ( new PointCloudType );
    
    ifstream fin ( fileName.c_str(), ios::in );
    fin.ignore ( 1024, '\n' );
    
    uint32_t idx = 0;
    while ( ! fin.eof() ) {
        int32_t classType;
        double x, y;
        string field;
        vector<string> tokens;
        
        fin >> classType;
        fin >> field;
        
//         cout << classType << ", " << field << endl;
        
        boost::split ( tokens, field, boost::is_any_of ( ":" ) );
        
        std::stringstream ss1;
        ss1 << tokens[1];
        ss1 >> x;
        
        fin >> field;
        
        boost::split ( tokens, field, boost::is_any_of ( ":" ) );
        
        std::stringstream ss2;
        ss2 << tokens[1];
        ss2 >> y;
        
//         cout << classType << " (" << x << ", " << y << ")" << endl;
        
        PointType p;
        p.x = x;
        p.y = y;
        p.z = p.r = p.g = p.b = 0.0;
        
        if ( classType == 1 ) {
            p.r = 255;
            X->push_back ( p );
        } else {
            p.g = 255;
            Y->push_back ( p );
        }       
        fin.ignore ( 1024, '\n' );
    }
    
    fin.close();
}

void SVMPathPlanning::visualizeClasses ( const PointCloudType::Ptr & X, const PointCloudType::Ptr & Y )
{
    
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer ( new pcl::visualization::PCLVisualizer ( "3D Viewer" ) );
    viewer->setBackgroundColor ( 0, 0, 0 );
    viewer->initCameraParameters();
    viewer->addCoordinateSystem();
    
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgbX ( X );
    viewer->addPointCloud<PointType> ( X, rgbX, "X" );
    
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgbY ( Y );
    viewer->addPointCloud<PointType> ( Y, rgbY, "Y" );
    
    while ( ! viewer->wasStopped () ) {
        viewer->spinOnce();
    }
}

void SVMPathPlanning::getBorderFromPointClouds ( const PointCloudType::Ptr & X, const PointCloudType::Ptr & Y,
                                CornerLimitsType & minCorner, CornerLimitsType & maxCorner,
                                double resolution ) {
    
    m_problem.l = X->size() + Y->size();
    m_problem.y = new double[m_problem.l];
    m_problem.x = new svm_node[m_problem.l];
    
    cout << "L = " << m_problem.l << endl;
    
    cv::Mat mapViz = cv::Mat::ones (resolution, resolution, CV_8UC3);
    
    for ( uint32_t i = 0; i < X->size(); i++ ) {
        const PointType & p = X->at ( i );
        double x = p.x; //( p.x - minCorner.first );
        double y = p.y; //( p.y - minCorner.second );
        
        m_problem.y[i] = 1;
        m_problem.x[i].dim = NDIMS;
        m_problem.x[i].values = new double[NDIMS];
        m_problem.x[i].values[0] = x;
        m_problem.x[i].values[1] = y;
    }
    
    for ( uint32_t i = 0; i < Y->size(); i++ ) {
        const PointType & p = Y->at ( i );
        double x = p.x; //( p.x - minCorner.first );
        double y = p.y; //( p.y - minCorner.second );
        
//         cout << "1: ( " << x << ", " << y << ")" << endl;
        
        const int & idx = i + X->size();
        m_problem.y[idx] = 2;
        m_problem.x[idx].dim = NDIMS;
        m_problem.x[idx].values = new double[NDIMS];
        m_problem.x[idx].values[0] = x;
        m_problem.x[idx].values[1] = y;
    }
    
    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start);

    svm_model * model = svm_train(&m_problem, &m_param);

    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    std::cout << "Elapsed time = " << elapsed << endl;
    
    getContoursFromSVMPrediction((const svm_model*&)model, 0.005);
    
    // Visualization
    svm_node node;
//     node.dim = NDIMS;
//     node.values = new double[NDIMS];
// 
//     CornerLimitsType interval = make_pair<double, double>(
//         (maxCorner.first - minCorner.first),
//         (maxCorner.second - minCorner.second)
//     );
//     for (uint32_t y = 0; y < resolution; y++) {
//         double prevVal = DBL_MIN;
//         for (uint32_t x = 0; x < resolution; x++) {
//             node.values[0] = (interval.first * x / resolution) + minCorner.first;
//             node.values[1] = (interval.second * y / resolution) + minCorner.second;
//             
//             double val = svm_predict(model, &node);
//             
//             if ((prevVal != DBL_MIN) && (prevVal != val)) {                
//                 cv::Vec3b& elem = mapViz.at<cv::Vec3b> (y, x);
//                 elem[0] = 255;
//                 elem[1] = 255;
//                 elem[2] = 255;
//             }
//             prevVal = val;
//         }
//     }
//     
//     for (uint32_t i = 0; i < model->l; i++) {
//         double x = (model->SV[i].values[0] - minCorner.first) * resolution / interval.first;
//         double y = (model->SV[i].values[1] - minCorner.second) * resolution / interval.second;
//         
//         cv::Vec3b& elem = mapViz.at<cv::Vec3b> (y, x);
//         elem[0] = 0;
//         elem[1] = 0;
//         elem[2] = 255;
//     }
    
    
    
//     svm_free_and_destroy_model(&model);
//     for (uint32_t i = 0; i < m_problem.l; i++) {
//         delete m_problem.x[i].values;
//     }
//     delete [] node.values;
//     delete [] m_problem.x;
//     delete [] m_problem.y;
    
    cv::imshow ( "Map", mapViz );
    
    cv::waitKey ( 0 );
    
}

void SVMPathPlanning::reducePointCloud(PointCloudType::Ptr& pointCloud)
{
    pcl::KdTreeFLANN<PointType> kdtree;
    
    PointCloudType::Ptr tmpPointCloud(new PointCloudType);
    tmpPointCloud->reserve(pointCloud->size());
    tmpPointCloud->push_back(pointCloud->at(0));
//     kdtree.setPointRepresentation(boost::make_shared<const XYZRGBRepresentation> (pointRepresentation));
    
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);
    
    PointCloudType::iterator it;
    for (it = pointCloud->begin(); it != pointCloud->end(); it++) {
        kdtree.setInputCloud (tmpPointCloud);
        uint32_t nPointsFound = kdtree.nearestKSearch (*it, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
        
        if ((nPointsFound == 0) || (pointNKNSquaredDistance[0] > m_minPointDistance))
            tmpPointCloud->push_back(*it);
    }
    
    pointCloud = tmpPointCloud;
}

void SVMPathPlanning::getContoursFromSVMPrediction(const svm_model * &model, const double & resolution) {
    
    CornerLimitsType minCorner = make_pair<double, double>(DBL_MAX, DBL_MAX);
    CornerLimitsType maxCorner = make_pair<double, double>(DBL_MIN, DBL_MIN);
    for (uint32_t i = 0; i < model->l; i++) {
        const double & x = model->SV[i].values[0];
        const double & y = model->SV[i].values[1];
        
        if (minCorner.first > x) minCorner.first = x;
        if (minCorner.second > y) minCorner.second = y;
        if (maxCorner.first < x) maxCorner.first = x;
        if (maxCorner.second < y) maxCorner.second = y;
    }
    CornerLimitsType interval = make_pair<double, double>(
        (maxCorner.first - minCorner.first),
        (maxCorner.second - minCorner.second)
    );
    
    cv::Size outputSize(interval.first / resolution + 1, interval.second / resolution + 1);
    
    cv::Mat predictMap(outputSize, CV_8UC1);

    struct timespec start, finish;
    double elapsed;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    launchSVMPrediction(model, minCorner.first, minCorner.second, interval.first, interval.second, 
                        outputSize.height, outputSize.width, resolution, predictMap.data);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    std::cout << "Elapsed time for prediction = " << elapsed << endl;
    
    vector<vector<cv::Point> > contours;
    cv::findContours(predictMap, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    
    cv::Mat contoursImg = cv::Mat::zeros(outputSize, CV_8UC3);
    for (vector<vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); it++) {
        for (vector<cv::Point>::iterator it2 = it->begin(); it2 != it->end(); it2++) {
            cv::Vec3b& elem = contoursImg.at<cv::Vec3b> (it2->y, it2->x);
            elem[0] = 0;
            elem[1] = 0;
            elem[2] = 255;
        }
    }
    
    cv::imshow("predictMap", predictMap);
    cv::imshow("contoursImg", contoursImg);
//     cv::imwrite("/home/nestor/Dropbox/projects/MultiSVMPathPlanning/predictMap.png", predictMap);
}