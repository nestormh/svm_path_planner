#include <iostream>

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

#include<opencv2/opencv.hpp>

#define MAP_BASE "/home/nestor/Dropbox/projects/MultiSVMPathPlanning/maps/parkingETSII1.pgm"

// Multiclass SVM in CUDA
// http://code.google.com/p/multisvm/source/checkout
// http://patternsonascreen.net/cuSVM.html

typedef pcl::PointXYZRGB SVMPointType;
typedef pcl::PointCloud<SVMPointType> SVMPointCloud;

void clusterize(const pcl::PointCloud<SVMPointType>::Ptr & pointCloud,
                const double & resolution, 
                std::vector< std::vector <SVMPointType> > & classes) {    
    
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<SVMPointType>::Ptr tree (new pcl::search::KdTree<SVMPointType>);
    tree->setInputCloud (pointCloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<SVMPointType> ec;
    ec.setClusterTolerance (2.0 / resolution);
    ec.setMinClusterSize (1);
    ec.setMaxClusterSize (INT_MAX);
    ec.setSearchMethod (tree);
    ec.setInputCloud (pointCloud);
    ec.extract (cluster_indices);

    classes.reserve(cluster_indices.size());
    
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); it++) {
        
        std::vector <SVMPointType> newClass;
        newClass.reserve(it->indices.size());
    
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end (); pit++) {
            SVMPointType point = pointCloud->points[*pit];
            
            newClass.push_back(point);
        }
        
        classes.push_back(newClass);
    }
}

void map2Classes(const cv::Mat & map, double resolution, 
                 std::vector< std::vector < SVMPointType > > & classes,
                 std::pair<double, double> & minCorner,
                 std::pair<double, double> & maxCorner) {

    SVMPointCloud::Ptr pointCloud(new SVMPointCloud);
    pointCloud->reserve(map.rows * map.cols);

    minCorner = std::make_pair<double, double>(DBL_MAX, DBL_MAX);
    maxCorner = std::make_pair<double, double>(DBL_MIN, DBL_MIN);
    for (uint32_t i = 0, idx = 0; i < map.rows; i++) {
        for (uint32_t j = 0; j < map.rows; j++, idx++) {
            SVMPointType point;
            if (map.data[idx] == 0) {
                point.x = j;
                point.y = i;
                point.z = 0.0;
                
//                 point.r = 255;
//                 point.g = 0;
//                 point.b = 0;
                
                if (point.x < minCorner.first)
                    minCorner.first = point.x;
                
                if (point.x > maxCorner.first)
                    maxCorner.first = point.x;

                if (point.y < minCorner.second)
                    minCorner.second = point.y;
                
                if (point.y > maxCorner.second)
                    maxCorner.second = point.y;
                
                pointCloud->push_back(point);
            }
        }
    }
    
    for (double i = minCorner.first; i < maxCorner.first; i ++) {
        SVMPointType point;
        
        point.x = i;
        point.y = minCorner.second;
        point.z = 0.0;
        
        pointCloud->push_back(point);
        
        point.y = maxCorner.second;
        
        pointCloud->push_back(point);

    }

    for (double i = minCorner.second; i < maxCorner.second; i ++) {
        SVMPointType point;
        
        point.x = minCorner.first;
        point.y = i;
        point.z = 0.0;
        
        pointCloud->push_back(point);
        
        point.x = maxCorner.first;
        
        pointCloud->push_back(point);

    }
    
    clusterize(pointCloud, resolution, classes);
}

void visualizeClasses(const std::vector< std::vector < SVMPointType > > & classes) {
    SVMPointCloud::Ptr pointCloud(new SVMPointCloud);
    
    for (uint32_t i = 0; i < classes.size(); i++) {
        std::vector < SVMPointType > currentClass = classes[i];
        
        uchar color[] = { rand() & 255, rand() & 255, rand() & 255 };
        
        for (uint32_t j = 0; j < currentClass.size(); j++) {
            
            SVMPointType point = currentClass[j];
            
            point.r = color[0];
            point.g = color[1];
            point.b = color[2];
        
            pointCloud->push_back(point);
        }
    }
    
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->initCameraParameters();
    viewer->addCoordinateSystem();

    pcl::visualization::PointCloudColorHandlerRGBField<SVMPointType> rgb(pointCloud);
    viewer->addPointCloud<SVMPointType> (pointCloud, rgb, "pointCloud");
    
    while (! viewer->wasStopped ()) {    
        viewer->spinOnce();       
    }

    
}


int main(int argc, char **argv) {
    double resolution = 0.1;
    
    
    cv::Mat map = cv::imread(MAP_BASE, 0);
    
//     cv::imshow("Map", map);
//     
//     cv::waitKey(0);
    
    std::vector< std::vector < SVMPointType > > classes;
    std::pair<double, double> minCorner, maxCorner;
    
    map2Classes(map, 0.1, classes, minCorner, maxCorner);
    visualizeClasses(classes);
    
    return 0;
}
