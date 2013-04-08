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
#include <vector_functions.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/surface/gp3.h>

using namespace svmpp;

SVMPathPlanning::SVMPathPlanning()
{
    // default values
    m_param.svm_type = C_SVC;
    m_param.kernel_type = RBF;
    m_param.degree = 3;
    m_param.gamma = 150;//300;        // 1/num_features
    m_param.coef0 = 0;
    m_param.nu = 0.5;
    m_param.cache_size = 100;
    m_param.C = 10000;//500;
    m_param.eps = 1e-2;
    m_param.p = 0.1;
    m_param.shrinking = 0;
    m_param.probability = 0;
    m_param.nr_weight = 0;
    m_param.weight_label = NULL;
    m_param.weight = NULL;
//     cross_validation = 0;
    
    m_minPointDistance = 2.0;
    m_gridSize = cv::Size(300, 300);
    m_minDistBetweenObstacles = 2.5;

    m_existingNodes = PointCloudType::Ptr(new PointCloudType);
    
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
    
    m_minPointDistance = 0.0005;
    m_minDistBetweenObstacles = 0.005;
    
    loadDataFromFile ( "/home/nestor/Dropbox/projects/MultiSVMPathPlanning/out1.txt", X, Y );
    
    m_minCorner = make_double2(0,0);
    m_maxCorner = make_double2(1,1);
    
    getBorderFromPointClouds ( X, Y );
    
    cv::waitKey(0);
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
//         p.z = p.r = p.g = p.b = 0.0;
        
//         if ( classType == 1 ) {
//             p.r = 255;
//             X->push_back ( p );
//         } else {
//             p.g = 255;
//             Y->push_back ( p );
//         }       
        fin.ignore ( 1024, '\n' );
    }
    
    fin.close();
}

void SVMPathPlanning::clusterize(const PointCloudTypeExt::Ptr & pointCloud,
                                 std::vector< PointCloudType::Ptr > & classes) {    
    
//     pcl::KdTreeFLANN<PointType> kdtree;    
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);
    
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<PointTypeExt>::Ptr tree (new pcl::search::KdTree<PointTypeExt>);
    tree->setInputCloud (pointCloud);
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointTypeExt> ec;
    ec.setClusterTolerance (m_minDistBetweenObstacles);
    ec.setMinClusterSize (1);
    ec.setMaxClusterSize (INT_MAX);
    ec.setSearchMethod (tree);
    ec.setInputCloud (pointCloud);
    ec.extract (cluster_indices);
    
    pcl::search::KdTree<PointType>::Ptr tree2 (new pcl::search::KdTree<PointType>);
    
    classes.reserve(cluster_indices.size());
    
    m_minCorner = make_double2(DBL_MAX, DBL_MAX);
    m_maxCorner = make_double2(DBL_MIN, DBL_MIN);
    
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); it++) {
        
        PointCloudType::Ptr newClass(new PointCloudType);
        newClass->reserve(it->indices.size());
        
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end (); pit++) {
            PointType point;
            point.x = pointCloud->points[*pit].x;
            point.y = pointCloud->points[*pit].y;
            
            uint32_t nPointsFound = 0;
            
            if (pit != it->indices.begin()) {
                tree2->setInputCloud (newClass);
                nPointsFound = tree2->nearestKSearch (point, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
            }
            
            if ((nPointsFound == 0) || (pointNKNSquaredDistance[0] > m_minPointDistance)) {
                if (point.x < m_minCorner.x) m_minCorner.x = point.x;
                if (point.x > m_maxCorner.x) m_maxCorner.x = point.x;
                if (point.y < m_minCorner.y) m_minCorner.y = point.y;
                if (point.y > m_maxCorner.y) m_maxCorner.y = point.y;
                
                newClass->push_back(point);
            }
        }
        
        classes.push_back(newClass);
    }

    m_maxCorner.x = 1.1 * m_maxCorner.x;
    m_maxCorner.y = 1.1 * m_maxCorner.y;
    m_minCorner.x = 0.9 * m_minCorner.x;
    m_minCorner.y = 0.9 * m_minCorner.y;
    
}

void SVMPathPlanning::addLineToPointCloud(const PointType& p1, const PointType& p2, 
                                          const uint8_t & r, const uint8_t & g, const uint8_t  & b,
                                          PointCloudTypeExt::Ptr & linesPointCloud) {
    
    double dist = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
    
    const uint32_t nSamples = (uint32_t)(ceil(dist / 0.02));
    
    for (uint32_t i = 0; i <= nSamples; i++) {
        pcl::PointXYZRGB p;
        p.x = p1.x + ((double)i / nSamples) * (p2.x - p1.x);
        p.y = p1.y + ((double)i / nSamples) * (p2.y - p1.y);
        p.z = p1.z + ((double)i / nSamples) * (p2.z - p1.z);
        
        p.r = r;
        p.g = g;
        p.b = b;
        
        linesPointCloud->push_back(p);
    } 
}

void SVMPathPlanning::visualizeClasses(const std::vector< PointCloudType::Ptr > & classes) {
                      
    PointCloudTypeExt::Ptr pointCloud(new PointCloudTypeExt);
    
    for (uint32_t i = 0; i < classes.size(); i++) {
        PointCloudType::Ptr currentClass = classes[i];
        
        uchar color[] = { rand() & 255, rand() & 255, rand() & 255 };
        
        for (uint32_t j = 0; j < currentClass->size(); j++) {
            
            PointTypeExt point;
            
            point.x = currentClass->at(j).x;
            point.y = currentClass->at(j).y;
            point.z = 0.0;
            point.r = color[0];
            point.g = color[1];
            point.b = color[2];
            
            pointCloud->push_back(point);
        }
    }
    
    PointCloudTypeExt::Ptr trajectory(new PointCloudTypeExt);
    trajectory->reserve(m_existingNodes->size());
    
    for (uint32_t i = 0; i < m_existingNodes->size(); i++) {
        PointTypeExt point;
        
        point.x = m_existingNodes->at(i).x;
        point.y = m_existingNodes->at(i).y;
        point.z = -1.0;
        point.r = 0;
        point.g = 255;
        point.b = 0;
        
        trajectory->push_back(point);
    }
    
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->initCameraParameters();
    viewer->addCoordinateSystem();
    
//     pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgb(pointCloud);
//     viewer->addPointCloud<PointTypeExt> (pointCloud, rgb, "pointCloud");
    
    pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbTrajectory(trajectory);
    viewer->addPointCloud<PointTypeExt> (trajectory, rgbTrajectory, "trajectory");
    
    PointCloudTypeExt::Ptr linesPointCloud(new PointCloudTypeExt);    
    for (vector< pair<uint32_t, uint32_t> >::iterator it = m_matches.begin(); it != m_matches.end(); it++) {
        addLineToPointCloud(m_existingNodes->at(it->first), m_existingNodes->at(it->second), 255, 0, 0, linesPointCloud);
    }
    
    pcl::visualization::PointCloudColorHandlerRGBField<PointTypeExt> rgbTriangulation(linesPointCloud);
    viewer->addPointCloud<PointTypeExt> (linesPointCloud, rgbTriangulation, "linesPointCloud");
    
    while (! viewer->wasStopped ()) {    
        viewer->spinOnce();       
    }
    
    
}

void SVMPathPlanning::getBorderFromPointClouds (PointCloudType::Ptr & X, PointCloudType::Ptr & Y ) {
    
    CornerLimitsType interval = make_double2(m_maxCorner.x - m_minCorner.x, 
                                             m_maxCorner.y - m_minCorner.y);
    
    m_distBetweenSamples = 1.5 * sqrt((interval.x / m_gridSize.width) * (interval.x / m_gridSize.width) +
                                      (interval.y / m_gridSize.height) * (interval.y / m_gridSize.height));
    
    m_problem.l = X->size() + Y->size();
    m_problem.y = new double[m_problem.l];
    m_problem.x = new svm_node[m_problem.l];
    
#ifdef DEBUG
    cout << "L = " << m_problem.l << endl;
#endif
    
    for ( uint32_t i = 0; i < X->size(); i++ ) {
        const PointType & p = X->at ( i );
        
        double x = ( p.x - m_minCorner.x ) / interval.x;
        double y = ( p.y - m_minCorner.y ) / interval.y;
        
        m_problem.y[i] = 1;
        m_problem.x[i].dim = NDIMS;
        m_problem.x[i].values = new double[NDIMS];
        m_problem.x[i].values[0] = x;
        m_problem.x[i].values[1] = y;
    }
    
    for ( uint32_t i = 0; i < Y->size(); i++ ) {
        const PointType & p = Y->at ( i );
        double x = (p.x - m_minCorner.x ) / interval.x;
        double y = ( p.y - m_minCorner.y ) / interval.y;
        
        const int & idx = i + X->size();
        m_problem.y[idx] = 2;
        m_problem.x[idx].dim = NDIMS;
        m_problem.x[idx].values = new double[NDIMS];
        m_problem.x[idx].values[0] = x;
        m_problem.x[idx].values[1] = y;
    }
   
#ifdef DEBUG
    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    
    svm_model * model = svm_train(&m_problem, &m_param);

#ifdef DEBUG
    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    std::cout << "Elapsed time = " << elapsed << endl;
#endif    

    getContoursFromSVMPrediction((const svm_model*&)model, interval);

    svm_free_and_destroy_model(&model);
    for (uint32_t i = 0; i < m_problem.l; i++) {
        delete m_problem.x[i].values;
    }
    delete [] m_problem.x;
    delete [] m_problem.y;
}

inline void SVMPathPlanning::getContoursFromSVMPrediction(const svm_model * &model, const CornerLimitsType & interval) {
    
    cv::Mat predictMap(m_gridSize, CV_8UC1);

#ifdef DEBUG
    struct timespec start, finish;
    double elapsed;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    
    launchSVMPrediction(model, m_gridSize.height, m_gridSize.width, predictMap.data);
    
#ifdef DEBUG
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    std::cout << "Elapsed time for prediction = " << elapsed << endl;
#endif
   
    vector<vector<cv::Point> > contours;
    cv::findContours(predictMap, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
            
    for (vector<vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); it++) {
        for (vector<cv::Point>::iterator it2 = it->begin(); it2 != it->end(); it2++) {
            PointType point;
            point.x = (it2->x * interval.x / m_gridSize.width) + m_minCorner.x;
            point.y = (it2->y * interval.y / m_gridSize.height) + m_minCorner.y;
            
            if ((point.x >= m_minCorner.x / 0.9) && (point.x <= m_maxCorner.x / 1.1) &&
                (point.y >= m_minCorner.y / 0.9) && (point.y <= m_maxCorner.y / 1.1)) {
            
                m_existingNodes->push_back(point);
            }
        }
    }
}

// TODO: aprovechar esta parte para hacer limpieza, si es necesario ¿o meqjor al añadirlas al grafo inicial?
void SVMPathPlanning::generateRNG() {
    
    // As we are working in 2D, we do not need normal estimation
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    normals->resize(m_existingNodes->size());

    // Concatenate the XYZ and normal fields
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields (*m_existingNodes, *normals, *cloud_with_normals);
    
    // Create search tree
    pcl::search::KdTree<pcl::PointNormal>::Ptr treeNormal (new pcl::search::KdTree<pcl::PointNormal>);
    treeNormal->setInputCloud (cloud_with_normals);
    
    // Initialize objects
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> triangulator;
    pcl::PolygonMesh triangles;
    
    // Set the maximum distance between connected points (maximum edge length)
    triangulator.setSearchRadius (2.0);
    
    // Set typical values for the parameters
    triangulator.setMu (2.0);
    triangulator.setMaximumNearestNeighbors (50);
    triangulator.setMaximumSurfaceAngle(M_PI); // 360 degrees
    triangulator.setMinimumAngle(0); // 360 degrees
    triangulator.setMaximumAngle(M_PI); // 360 degrees
    triangulator.setNormalConsistency(false);
    triangulator.setConsistentVertexOrdering (true);
    
    // Get result
    triangulator.setInputCloud (cloud_with_normals);
    triangulator.setSearchMethod (treeNormal);
    triangulator.reconstruct (triangles);
    
    for (vector<pcl::Vertices>::iterator it = triangles.polygons.begin(); it != triangles.polygons.end(); it++) {
        for (uint32_t i = 0; i < it->vertices.size(); i++) {
            m_matches.push_back(make_pair<uint32_t, uint32_t>(it->vertices[i], it->vertices[(i + 1) % it->vertices.size()]));
        }
    }
    
    // Graph is completed with lines that not form a polygon
    vector<int> pointIdxNKNSearch;
    vector<float> pointNKNSquaredDistance;
    pcl::KdTreeFLANN<PointType>::Ptr treeNNG (new pcl::KdTreeFLANN<PointType>);
    treeNNG->setInputCloud (m_existingNodes);
    treeNNG->setSortedResults(true);
    
    for (uint32_t i = 0; i < m_existingNodes->size(); i++) {
        treeNNG->radiusSearch(m_existingNodes->at(i), m_distBetweenSamples, pointIdxNKNSearch, pointNKNSquaredDistance);
        for (uint32_t j = 0; j < pointIdxNKNSearch.size(); j++) {
            m_matches.push_back(make_pair<uint32_t, uint32_t>(i, pointIdxNKNSearch[j]));
        }
    }
}

void SVMPathPlanning::obtainGraphFromMap(const PointCloudTypeExt::Ptr & inputCloud)
{
    struct timespec start, finish;
    double elapsed;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
        
    vector< PointCloudType::Ptr > classes;
    clusterize(inputCloud, classes);
    
    vector< PointCloudType::Ptr >::iterator it1, it2;
    
    for (it1 = classes.begin(); it1 != classes.end(); it1++) {
        PointCloudType::Ptr pointCloud1(new PointCloudType);
        *pointCloud1 = *(*it1);
        PointCloudType::Ptr pointCloud2(new PointCloudType);
        for (it2 = classes.begin(); it2 != classes.end(); it2++) {
            if (it1 != it2) {
                *pointCloud2 += *(*it2);
            }
        }
                                         
        getBorderFromPointClouds (pointCloud1, pointCloud2);
//         break;
    }
    generateRNG();
    
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    std::cout << "Total time = " << elapsed << endl;
    
    visualizeClasses(classes);
}
