// #include <iostream>
// #include <string>
// #include <vector>
// #include <iostream>
// #include <fstream>
// 
// #include <boost/date_time/posix_time/posix_time_types.hpp>
// #include <boost/date_time/posix_time/posix_time.hpp>
// 
// #include <pcl/visualization/point_picking_event.h>
// 
// #include <opencv2/opencv.hpp>
// 
// #include "fileList.h"
// #include "ccpdtracking.h"
// 
// inline void readCameraParams(std::ifstream & fin, t_Camera_params & params) {
//     std::string paramsName;    
//     fin >> paramsName;
//     fin >> params.width;
//     fin >> params.height;
//     fin >> params.ku;
//     fin >> params.kv;
//     fin >> params.u0;
//     fin >> params.v0;
//     fin >> params.x;
//     fin >> params.y;
//     fin >> params.z;
//     fin >> params.yaw;
//     fin >> params.pitch;
//     fin >> params.roll;
// }
// 
// void setParamsGeometry(t_Camera_params & params, const int & width, const int & height) {
// 
//     // Modificamos el centro
//     if (params.u0 != 0.0) {
//         params.u0 *= (double) width / (double) params.width;
//     } else {
//         params.u0 = (double) width / 2.0;
//     }
// 
//     if (params.v0 != 0.0) {
//         params.v0 *= (double) height / (double) params.height;
//     } else {
//         params.v0 = (double) height / 2.0;
//     }
// 
//     // Modificamos la focal
//     params.ku *= (double) width / (double) params.width;
//     params.kv *= (double) height / (double) params.height;
//     
//     params.y *= (double) width / (double) params.width;    
//     params.z *= (double) height / (double) params.height;
// 
//     params.width = width;
//     params.height = height;
// }
// 
// inline std::vector<t_ImageNames> readListFromFile(std::string basePath, std::string sequenceName, 
//                                            t_Camera_params & paramsLeft, t_Camera_params &paramsCenter,
//                                            t_Camera_params & paramsRight) {
//     
//     std::string fileName = basePath + "/" + sequenceName + "/" + sequenceName + ".txt";
// 
//     std::vector<t_ImageNames> imageList;
//     
//     std::ifstream fin(fileName.c_str(), std::ios::in);
//     readCameraParams(fin, paramsLeft);
//     readCameraParams(fin, paramsCenter);
//     readCameraParams(fin, paramsRight);
//     while (fin.good()) {
//         t_ImageNames imageNames;
//         
//         std::string str;
//         fin >> str;
//         std::stringstream ss(str);
//         boost::posix_time::time_duration td;
//         ss >> imageNames.timestamp;
//         
//         fin >> imageNames.LW;
//         fin >> imageNames.LS;
//         fin >> imageNames.RW;
//         fin >> imageNames.RS;
//         
//         imageList.push_back(imageNames);
//     }
//     fin.close();
//     
//     return imageList;
// }
// 
// void mouseEventOccurred (const pcl::visualization::MouseEvent &event,
//                          void* viewer_void) {
//     
//   boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
//   if (event.getButton () == pcl::visualization::MouseEvent::LeftButton &&
//       event.getType () == pcl::visualization::MouseEvent::MouseButtonRelease)      
//   {
//     std::cout << "Left mouse button released at position (" << event.getX () << ", " << event.getY () << ")" << std::endl;
// 
//     char str[512];
//     sprintf (str, "texto");
//     viewer->addText ("clicked here", event.getX (), event.getY (), str);
//   }
// }
// 
// void point_picking_callback (const pcl::visualization::PointPickingEvent& event, void* cookie) {     
//     float x, y, z;
//     event.getPoint(x, y, z);
//     std::cout << "(" << x << ", " << y << ", " << z << ")" << std::endl;
// } 
// 
// 
// int main(int argc, char **argv) {    
//     // TODO: Convertir en params
//     std::string basePath = "/local/imaged/calibrated";
//     std::string sequenceName = "stereocalib-001";
//     int idx = 139;
//     bool useShortBaseline = true;
//     bool output = true;
//     bool stopOnFrames = true;
//     t_SGBM_params stereoParams = { 0, 64, 3, 3 * 3 * 4, 3 * 3 * 32, 1, 63, 10, 100, 32, true };    
//     bool useDispImg = false;    
//     t_CPD_opts optsCPD = { "nonrigid_lowrank", true, true, 150, 1e-5, false, 0.1, 2, false, false, 2, 3, 3, 0.01, true, basePath, sequenceName, 0 };
//     bool loadMatchesFromFile = false;
//     double groundThresh = 0.2;
//     double bgThresh = 20;
//     // TODO: Hasta aqui
//     
//     cv::namedWindow("LEFT", CV_WINDOW_AUTOSIZE | CV_GUI_EXPANDED);
//     cv::namedWindow("RIGHT", CV_WINDOW_AUTOSIZE | CV_GUI_EXPANDED);
//         
//     cv::Mat left, right;
//     
//     t_Camera_params paramsLeft;
//     t_Camera_params paramsCenter;
//     t_Camera_params paramsRight;
//     
//     std::vector<t_ImageNames> imageList = readListFromFile(basePath, sequenceName, paramsLeft, paramsCenter, paramsRight);
//     
//     if (imageList.size() == 0) {
//         std::cout << "No se han encontrado imágenes en la ruta especificada. ¿Estás seguro de que montaste el disco?" << std::endl;        
//         exit(0);
//     }
//     
//     // Creamos el visualizador PCL
//     boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//     viewer->setBackgroundColor (0, 0, 0);
//     viewer->initCameraParameters();
//     viewer->addCoordinateSystem();
// //     viewer->registerMouseCallback (mouseEventOccurred, (void*)&viewer);
//     std::string ppMsg3D ("PP"); 
//     viewer->registerPointPickingCallback(&point_picking_callback); //, *this, (void*)(&ppMsg3D));
// //     viewer->spinOnce(1);
// //     std::vector< pcl::visualization::Camera > cameras;
// //     viewer->getCameras(cameras);
// //     cameras[0].pos[0] = 0.310498;
// //     cameras[0].pos[1] = 0.433701;
// //     cameras[0].pos[2] = -1.23387;
// //     cameras[0].view[0] = 0.0135085;
// //     cameras[0].view[1] = -0.954325;
// //     cameras[0].view[2] = -0.298466;
//     
//     CCPDTracking cpd(stereoParams, viewer, useDispImg, output, loadMatchesFromFile);
//     
//     int waitTime = 100;
//     
//     for (uint32_t i = idx; i < imageList.size(); i++) {
//         std::cout << "Iteración : " << i << std::endl;
//         t_Camera_params paramsL, paramsR;
//         if (useShortBaseline) {
//             left = cv::imread(basePath + "/" + sequenceName + "/" + imageList.at(i).LS);
//             right = cv::imread(basePath + "/" + sequenceName + "/" + imageList.at(i).RS);
//             paramsL = paramsCenter;
//             paramsR = paramsRight;
//         } else {
//             left = cv::imread(basePath + "/" + sequenceName + "/" + imageList.at(i).LW);
//             right = cv::imread(basePath + "/" + sequenceName + "/" + imageList.at(i).RW);
//             paramsL = paramsLeft;
//             paramsR = paramsRight;
//         }
//         
//         setParamsGeometry(paramsL, left.cols, left.rows);
//         setParamsGeometry(paramsR, right.cols, right.rows);
//         
//         optsCPD.iteration = i;
//         cpd.setCPDOptions(optsCPD);
//         
//         cpd.update(left, right, groundThresh, bgThresh, paramsL, paramsR);
//                 
//         if (output) {
// //             cv::imshow("LEFT", left);
// //             cv::imshow("RIGHT", right);
//             
//             bool doLoop = stopOnFrames;
//             while (doLoop) {
//                 viewer->spinOnce(1);
// //                 for (uint32_t i = 0; i < cameras.size(); i++) {
// //                     double * pos = cameras[i].pos;
// //                     double * view = cameras[i].view;
// //                     std::cout << i << "-> Pos = " << pos[0] << ", " << pos[1] << ", " << pos[2] << std::endl;
// //                     std::cout << i << "-> view = " << view[0] << ", " << view[1] << ", " << view[2] << std::endl;
// //                 }
//                 int ret = cv::waitKey(100);
//                 switch (ret) {
//                     case 1048603:       // ESC
//                         exit(0);
//                         break;   
//                     case 1048608:       // SPACE
//                         stopOnFrames = !stopOnFrames;
//                         doLoop = stopOnFrames;
//                         break;   
//                     case 1048679: {     // g
//                         uint32_t last = i;
//                         std::cout << "Introduzca frame destino (Actual: " << i << ", Max: " << imageList.size() - 1 << std::endl;
//                         std::cin >> i;
//                         if ((i < 0) || (i > imageList.size())) {
//                             std::cout << "Frame inválido" << std::endl;
//                             i = last;
//                         }
//                         break;
//                     }
//                     case 1048686:
//                         doLoop = false;
//                         break;
//                     case -1:
//                         break;
//                     default:
//                         std::cout << "Otra tecla:" << ret << std::endl;                        
//                         ;
//                 }                
//             }
//         }
//             
//     }
//     
//     
// //     const char* ops = "-nojvm"; 
// //     std::cout << 1 << std::endl;
// //     mclInitializeApplication(&ops, 1);    
// //     std::cout << 2 << std::endl;
// //     libHelloMatlabInitialize();
// //     std::cout << 3 << std::endl;
// // //     mwArray arr;
// // //     hellomatlab();
// //     std::cout << 4 << std::endl;
// //     libHelloMatlabTerminate();
// //     std::cout << 5 << std::endl;
// //     mclTerminateApplication(); 
// //     std::cout << 6 << std::endl;
// 
//     
//     
//     return 0;
// }

#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

boost::shared_ptr<pcl::visualization::PCLVisualizer>
simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  //viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

int
main(int argc, char** argv)
{
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
//     if (pcl::console::find_argument (argc, argv, "-s") >= 0 || pcl::console::find_argument (argc, argv, "-sf") >= 0)
//     {
//       cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0);
//       cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0);
//       if (i % 5 == 0)
//         cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0);
//       else if(i % 2 == 0)
//         cloud->points[i].z =  sqrt( 1 - (cloud->points[i].x * cloud->points[i].x)
//                                       - (cloud->points[i].y * cloud->points[i].y));
//       else
//         cloud->points[i].z =  - sqrt( 1 - (cloud->points[i].x * cloud->points[i].x)
//                                         - (cloud->points[i].y * cloud->points[i].y));
//     }
//     else
//     {
      cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0);
      cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0);
      if( i % 2 == 0)
        cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0);
      else
        cloud->points[i].z = -1 * (cloud->points[i].x + cloud->points[i].y);
//     }
  }

  std::vector<int> inliers;

  // created RandomSampleConsensus object and compute the appropriated model
  pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr
    model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZ> (cloud));
  pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr
    model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud));
//   if(pcl::console::find_argument (argc, argv, "-f") >= 0)
//   {
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_p);
    ransac.setDistanceThreshold (.01);
    ransac.computeModel();
    ransac.getInliers(inliers);
//   }
//   else if (pcl::console::find_argument (argc, argv, "-sf") >= 0 )
//   {
//     pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_s);
//     ransac.setDistanceThreshold (.01);
//     ransac.computeModel();
//     ransac.getInliers(inliers);
//   }

  // copies all inliers of the model computed to another PointCloud
  pcl::copyPointCloud<pcl::PointXYZ>(*cloud, inliers, *final);

  // creates the visualization object and adds either our orignial cloud or all of the inliers
  // depending on the command line arguments specified.
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  if (pcl::console::find_argument (argc, argv, "-f") >= 0 || pcl::console::find_argument (argc, argv, "-sf") >= 0)
//     viewer = simpleVis(final);
//   else
//     viewer = simpleVis(cloud);
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
  return 0;
 }