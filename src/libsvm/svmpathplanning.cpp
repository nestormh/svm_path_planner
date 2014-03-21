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
    m_problem.x = new svm_node*[m_problem.l];
    
    cout << "L = " << m_problem.l << endl;
    
    cv::Mat mapViz = cv::Mat::ones (resolution, resolution, CV_8UC3);
    
    for ( uint32_t i = 0; i < X->size(); i++ ) {
        const PointType & p = X->at ( i );
        double x = p.x; //( p.x - minCorner.first );
        double y = p.y; //( p.y - minCorner.second );
        
        m_problem.y[i] = 1;
        m_problem.x[i] = new svm_node[3];
        m_problem.x[i][0].index = 1;
        m_problem.x[i][1].index = 2;
        m_problem.x[i][2].index = -1;
        
        m_problem.x[i][0].value = x;
        m_problem.x[i][1].value = y;
        m_problem.x[i][2].value = -1;
    }
    
    for ( uint32_t i = 0; i < Y->size(); i++ ) {
        const PointType & p = Y->at ( i );
        double x = p.x; //( p.x - minCorner.first );
        double y = p.y; //( p.y - minCorner.second );
        
//         cout << "1: ( " << x << ", " << y << ")" << endl;
        
        const int & idx = i + X->size();
        m_problem.y[idx] = 2;
        m_problem.x[idx] = new svm_node[3];
        m_problem.x[idx][0].index = 1;
        m_problem.x[idx][1].index = 2;
        m_problem.x[idx][2].index = -1;
        
        m_problem.x[idx][0].value = x;
        m_problem.x[idx][1].value = y;
        m_problem.x[idx][2].value = -1;

    }
    
    svm_model * model = svm_train(&m_problem, &m_param);
    
    // Visualization
    svm_node node[3];
    node[0].index = 1;
    node[1].index = 2;
    node[2].index = -1;

    CornerLimitsType interval = make_pair<double, double>(
        (maxCorner.first - minCorner.first),
        (maxCorner.second - minCorner.second)
    );
    for (uint32_t y = 0; y < resolution; y++) {
        double prevVal = DBL_MIN;
        for (uint32_t x = 0; x < resolution; x++) {
            node[0].value = (interval.first * x / resolution) + minCorner.first;
            node[1].value = (interval.second * y / resolution) + minCorner.second;
            
            double val = svm_predict(model, node);
            
            if ((prevVal != DBL_MIN) && (prevVal != val)) {                
                cv::Vec3b& elem = mapViz.at<cv::Vec3b> (y, x);
                elem[0] = 255;
                elem[1] = 255;
                elem[2] = 255;
            }
            prevVal = val;
        }
    }
    
    svm_free_and_destroy_model(&model);
    for (uint32_t i = 0; i < m_problem.l; i++) {
        delete m_problem.x[i];
    }
    delete [] m_problem.x;
    delete [] m_problem.y;
    
    cv::imshow ( "Map", mapViz );
    
    cv::waitKey ( 0 );
    
}

