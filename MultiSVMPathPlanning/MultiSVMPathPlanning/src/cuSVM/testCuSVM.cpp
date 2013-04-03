#include "cuSVM.h"

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

typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<PointType> PointCloudType;
typedef pair<double, double> CornerLimitsType;

void loadDataFromFile(const std::string & fileName,
                      PointCloudType::Ptr & X,
                      PointCloudType::Ptr & Y) {
        
    X = PointCloudType::Ptr(new PointCloudType);
    Y = PointCloudType::Ptr(new PointCloudType);
    
    ifstream fin(fileName.c_str(), ios::in);
    fin.ignore(1024, '\n');
    
    while (! fin.eof()) {
        int32_t classType;
        double x, y;
        string field;
        vector<string> tokens;
        
        fin >> classType;
        fin >> field;
        
        boost::split(tokens, field, boost::is_any_of(":"));
        
        std::stringstream ss1;
        ss1 << tokens[1];
        ss1 >> x;
        
        fin >> field;
        
        boost::split(tokens, field, boost::is_any_of(":"));
        
        std::stringstream ss2;
        ss2 << tokens[1];
        ss2 >> y;
        
        cout << classType << " (" << x << ", " << y << ")" << endl;
        
        PointType p;
        p.x = x;
        p.y = y;
        p.z = p.r = p.g = p.b = 0.0;
        
        if (classType == -1) {
            p.r = 255;
            X->push_back(p);
        } else {
            p.g = 255;
            Y->push_back(p);
        }
        fin.ignore(1024, '\n');
    }
    
    fin.close();
}

void visualizeClasses(const PointCloudType::Ptr & X, const PointCloudType::Ptr & Y) {
    
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->initCameraParameters();
    viewer->addCoordinateSystem();
    
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgbX(X);
    viewer->addPointCloud<PointType> (X, rgbX, "X");
    
    pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgbY(Y);
    viewer->addPointCloud<PointType> (Y, rgbY, "Y");
    
    while (! viewer->wasStopped ()) {    
        viewer->spinOnce();       
    }
}

void getBorderFromPointClouds(const PointCloudType::Ptr & X, const PointCloudType::Ptr & Y,
                              CornerLimitsType & minCorner, CornerLimitsType & maxCorner,
                              double resolution) {
    
    float tElapsed, beta;
    float * alpha;
    int m = X->size() + Y->size();
    int n = NDIMS;
    
    float * y = new float[m];
    float * x = new float[m * n];
    
    cv::Mat mapViz = cv::Mat::zeros((maxCorner.first - minCorner.first) / resolution,
                   (maxCorner.second - minCorner.second) / resolution,
                   CV_8UC3);
    
    for (uint32_t i = 0; i < X->size(); i++) {
        const PointType & p = X->at(i);
        int xPos = (p.x - minCorner.first) / resolution;
        int yPos = (p.y - minCorner.second) / resolution;

        cv::Vec3b& elem = mapViz.at<cv::Vec3b>(yPos, xPos);
        elem[2] = 255;
        
        y[i] = -1.0;
        x[i * n] = xPos;
        x[i * n + 1] = yPos;
    }
    
    for (uint32_t i = 0; i < Y->size(); i++) {
        const PointType & p = Y->at(i);
        int xPos = (p.x - minCorner.first) / resolution;
        int yPos = (p.y - minCorner.second) / resolution;
        
        cv::Vec3b& elem = mapViz.at<cv::Vec3b>(yPos, xPos);
        elem[1] = 255;
        
        const int & idx = i + X->size();
        y[idx] = 1;
        x[idx * n] = xPos;
        x[idx * n + 1] = yPos;
    }
    
    float C = 100.0;
    float gamma = 1.0 / m;
    float stoppingCrit = 0.001;
    
    SVMTrain(&tElapsed, alpha, &beta, y, x, C, gamma, m, n, stoppingCrit);
//     SVMTrain(float * elapsed, float *alpha,float *beta, float *y,float *x, float C, float kernelwidth, int m, int n, float StoppingCrit);
    
    cout << "tElapsed = " << tElapsed << endl;
    cout << "Beta = " << beta << endl;
    cout << "C = " << C << endl;
    cout << "gamma = " << gamma << endl;
    
//     uint32_t nSV = 0;
//     for (uint32_t i = 0; i < m; i++) {
//         std::cout << "testCuSVM " << __LINE__ << std::endl;
//         if (alpha[i] != 0) {
//             std::cout << "testCuSVM " << __LINE__ << std::endl;
//             cout << "( " << x[i * n] << ", " << x[i * n + 1] << ") = " << alpha[i] << endl;
//             
//             nSV++;
//         }
//         std::cout << "testCuSVM " << __LINE__ << std::endl;
//     }
//     cout << "Total SVs = " << nSV << endl;
    
    cv::imshow("Map", mapViz);
    
    cv::waitKey(0);
    
    delete x, y, alpha;
    
}

int main(int argc, char * argv[]) {
    
    PointCloudType::Ptr X, Y;
    
    loadDataFromFile("/home/nestor/Dropbox/projects/MultiSVMPathPlanning/ejemplo.txt", X, Y);
 
    CornerLimitsType minCorner = make_pair<double, double>(0, 0);
    CornerLimitsType maxCorner = make_pair<double, double>(200, 200);
    
    getBorderFromPointClouds(X, Y, minCorner, maxCorner, 1.0);  
    
//     visualizeClasses(X, Y);
    
    return 0;
}