/**
 * @file   gpusurf_engine.cpp
 * @author Paul Furgale and Chi Hay Tong
 * @date   Fri Apr 30 19:49:15 2010
 * $Rev: 3828 $
 * 
 * @brief  A command line utility and example usage of the GPUSURF library
 * 
 * 
 */


/*
Copyright (c) 2010, Paul Furgale and Chi Hay Tong
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are 
met:

* Redistributions of source code must retain the above copyright notice, 
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright 
  notice, this list of conditions and the following disclaimer in the 
  documentation and/or other materials provided with the distribution.
* The names of its contributors may not be used to endorse or promote 
  products derived from this software without specific prior written 
  permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <vector>
#include <boost/program_options.hpp>
#include <gpusurf/GpuSurfDetector.hpp>
#include "assert_macros.hpp"
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <opencv/cv.hpp>
#include <fstream>
#include "CudaSynchronizedMemory.hpp"
#include "gpu_utils.h"
#include "gpu_globals.h"
#include <opencv/highgui.h>
#include <opencv/highgui.hpp>

void drawKeypoints(std::vector<cv::KeyPoint> const & keypoints, cv::Mat & image)
{
  std::vector<cv::KeyPoint>::const_iterator k = keypoints.begin();
  cv::Scalar red(255,0,0);
  cv::Scalar blue(0,0,255);
  for( ; k != keypoints.end(); k++)
    {
      cv::Scalar * choice = NULL;
      if( isLastBitSet(k->response) )
	choice = &red;
      else
	choice = &blue;

      cv::Point2f dir(k->pt);
      float st = k->size * sin(k->angle);
      float ct = k->size * cos(k->angle);
      dir.x += ct;
      dir.y += st;
      cv::circle(image, k->pt, (int)k->size, *choice, 1,CV_AA);
      cv::line(image, k->pt, dir, *choice, 1, CV_AA);
    }
}

bool interestingStrengthIndex(std::pair<float,int> const & lhs, std::pair<float,int> const & rhs)
{
  return lhs.first > rhs.first;
}

std::string getFileBasename(std::string const & filename)
{
  size_t extIdx = filename.find_last_of('.');
  return  filename.substr(0,extIdx);
}

void load_keypoints(std::string const & inputKeypointFile, std::vector<cv::KeyPoint> & inKeypoints, int imRows, int imCols)
{
  std::ifstream fin(inputKeypointFile.c_str());
  ASRL_ASSERT(fin.good(),"Unable to open keypoint file " << inputKeypointFile << " for reading");


  std::string line;
  std::getline(fin,line);
  int i = 1;
  while(!fin.eof())
    {
      std::istringstream lin(line);
      cv::KeyPoint k;
      lin >> k.pt.x;
      lin >> k.pt.y;
      lin >> k.size;
      lin >> k.response;
      lin >> k.angle;
      lin >> k.octave;
      int laplacian;
      lin >> laplacian;
      if(laplacian == 1)
	{
	  setLastBit(k.response);
	}
      else
	{
	  clearLastBit(k.response);
	}
      //std::cout << "read kpt: " << k.pt.x << "," << k.pt.y << ", " << k.size << "," << k.angle << std::endl;
      ASRL_ASSERT_GE_LT(k.pt.x,0,imCols,"Keypoint " << i << " is out of bounds");
      ASRL_ASSERT_GE_LT(k.pt.y,0,imRows,"Keypoint " << i << " is out of bounds");
      ASRL_ASSERT_GE_LT(k.angle,-3.15,3.15,"Keypoint " << i << " angle is out of bounds");
      ASRL_ASSERT_GE_LT(k.size,1,100,"Keypoint " << i << " size is out of bounds");
      inKeypoints.push_back(k);
      std::getline(fin,line);
      i++;
    }
  //std::cout << "Loaded " << inKeypoints.size() << " keypoints\n";
}

void save_keypoints(std::vector<cv::KeyPoint> const & keypoints,
		    std::vector<float> const & descriptors,
		    std::vector<std::pair<float,int> > const & strengthIndex,
		    std::string const & imageFileName,
		    bool noDescriptor)
{
  std::string keypointFileName = getFileBasename(imageFileName) + "-gpusurf.key";
  std::ofstream fout(keypointFileName.c_str());

  //fout.precision(20);
  for(unsigned i = 0; i < strengthIndex.size() ; i++)
    {
      cv::KeyPoint const & k = keypoints[ strengthIndex[i].second ];
      fout << k.pt.x     << "\t"
	   << k.pt.y     << "\t"
	   << k.size     << "\t"
	   << k.response << "\t"
	   << k.angle    << "\t"
	   << k.octave   << "\t"
           << isLastBitSet(k.response) << "\t";

      if(! noDescriptor )
	{
	  std::vector<float>::const_iterator d = descriptors.begin() + (strengthIndex[i].second * ASRL_SURF_DESCRIPTOR_DIM);
	  for(int j = 0; j < ASRL_SURF_DESCRIPTOR_DIM; j++, d++)
	    {
	      fout << *d << "\t";
	    }
	}
      fout << std::endl;

    }
  fout.close();
}

namespace po = boost::program_options;

int gpuSurf(int argc, char * argv[]) {
  std::vector<std::string> images;

  try {

    asrl::GpuSurfConfiguration configuration;
    int targetFeatures;
    int detectorType;
    bool noDescriptor;
    bool noOrientation;
    int device = 0;
    std::string inputFile;
    int demoCamera = 0;

    // Initialize the program options.
    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "produce help message")
      ("image",po::value<std::vector<std::string> >(&images),"Image filename")
      ("threshold,t",po::value<float>(&configuration.threshold)->default_value(0.1f),"The interest operator threshold")
      ("n-octaves,v",po::value<int>(&configuration.nOctaves)->default_value(4),"The number of octaves")
      ("n-intervals,i", po::value<int>(&configuration.nIntervals)->default_value(4),"The number of intervals")
      ("initial-scale,s",po::value<float>(&configuration.initialScale)->default_value(2.f),"The initial scale of the detector")
      ("l1",po::value<float>(&configuration.l1)->default_value(3.f/1.5f),"Filter parameter l1")
      ("l2",po::value<float>(&configuration.l2)->default_value(5.f/1.5f),"Filter parameter l2")
      ("l3",po::value<float>(&configuration.l3)->default_value(3.f/1.5f),"Filter parameter l3")
      ("l4",po::value<float>(&configuration.l4)->default_value(1.f/1.5f),"Filter parameter l4")
      ("edge-scale",po::value<float>(&configuration.edgeScale)->default_value(0.81f),"The edge rejection mask scale")
      ("initial-step",po::value<int>(&configuration.initialStep)->default_value(1),"The initial step in pixels")
      ("n-features,n",po::value<int>(&targetFeatures)->default_value(0),"The target number of features to return")
      ("debug-iimg","dump the integral image to disk. The integral image will be saved as imagename-iimg.bin")
      ("no-descriptor","don't compute the descriptor")
      ("no-orientation","don't compute orientation")
      ("fast-orientation","use an orientation calculation that is slightly faster")
      ("device",po::value<int>(&device)->default_value(0),"The cuda device to use")
      ("input-keypoints",po::value<std::string>(&inputFile),"Read the keypoints from a file, push them to the GPU then produce the orientation and descriptor. If used together with --no-orientation, the input orientation will be used.")
      ("detector-threads-x",po::value<int>(&configuration.detector_threads_x)->default_value(8),"The number of threads to use in the interest point detector kernel (dimension 1)")
      ("detector-threads-y",po::value<int>(&configuration.detector_threads_y)->default_value(8),"The number of threads to use in the interest point detector kernel (dimension 2)")
      ("nonmax-threads-x",po::value<int>(&configuration.nonmax_threads_x)->default_value(8),"The number of threads to use in the non-max suppression kernel (dimension 1)")
      ("nonmax-threads-y",po::value<int>(&configuration.nonmax_threads_y)->default_value(8),"The number of threads to use in the non-max suppression kernel (dimension 2)")
      ("run-demo","Use OpenCV to run a demo using the camera.")
      ("demo-camera",po::value<int>(&demoCamera)->default_value(0),"The camera index to use when running the demo.")
      ;

    po::positional_options_description p;
    p.add("image", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
	      options(desc).positional(p).run(), vm);

    po::notify(vm);

    if (vm.count("help") || (vm.count("image") == 0 && ! vm.count("run-demo")) ) {
      std::cout << "Usage [options] image\n\n";
      std::cout << desc << "\n";
      return 1;
    }

    bool debugHessian = vm.count("debug-hessian") > 0;
    bool debugIimg = vm.count("debug-iimg") > 0;
    noDescriptor = vm.count("no-descriptor") > 0;
    noOrientation = vm.count("no-orientation") > 0;
    bool fastOrientation = vm.count("fast-orientation") > 0;

    

    cudaError_t err;
    int deviceCount;
    err = cudaGetDeviceCount(&deviceCount);
    ASRL_ASSERT_EQ(err,cudaSuccess, "Unable to get the CUDA device count: " << cudaGetErrorString(err));
    ASRL_ASSERT_GT(deviceCount,0,"There are no CUDA capable devices present");
    ASRL_ASSERT_GT(deviceCount,device,"Device index is out of bounds");
    err = cudaSetDevice(device);
    ASRL_ASSERT_EQ(err,cudaSuccess, "Unable to set the CUDA device: " << cudaGetErrorString(err));

    if(vm.count("run-demo") > 0)
      {
	asrl::GpuSurfDetector detector(configuration);
	cv::VideoCapture vc(demoCamera);
	cv::Mat image, greyImage;
	std::string windowName = "Speeded Up SURF";
	cv::namedWindow(windowName,CV_WINDOW_AUTOSIZE);
	int key = cv::waitKey(5);
	std::vector<cv::KeyPoint> keypoints;
	bool success = vc.grab();
	 
	while(success && key == -1)
	  {
	    success = vc.retrieve(image);
	    if(success)
	      {
		cv::cvtColor(image,greyImage,CV_BGR2GRAY);
		detector.buildIntegralImage(greyImage);
		detector.detectKeypoints();
		if(!noOrientation)
		  {
		    if(fastOrientation)
		      detector.findOrientationFast();
		    else
		      detector.findOrientation();
		  }
		detector.getKeypoints(keypoints);
		drawKeypoints(keypoints,image);
		cv::imshow(windowName, image);
		key = cv::waitKey(5);
		success = vc.grab();
	      }
	  }
      }
    else if(vm.count("input-keypoints") > 0)
      {
	ASRL_ASSERT_EQ(images.size(),1,"Exactly one image must be supplied with a keypoint file");
	// Read input keypoints from a file and compute the orientation and descriptor.

	asrl::GpuSurfDetector detector(configuration);
	cv::Mat cvImage;
	cvImage = cv::imread(images[0],CV_LOAD_IMAGE_GRAYSCALE);

	std::vector<cv::KeyPoint> inKeypoints;
	load_keypoints(inputFile,inKeypoints, cvImage.cols, cvImage.rows);
	detector.setKeypoints(inKeypoints);
	detector.buildIntegralImage(cvImage);
	if(! noOrientation )
	  {
	    if(fastOrientation)
	      detector.findOrientationFast();
	    else
	      detector.findOrientation();
	  }
	if(!noDescriptor)
	  {	
	    detector.computeDescriptors();
	  }
	std::vector<cv::KeyPoint> keypoints;
	std::vector<float> descriptors;
	detector.getKeypoints(keypoints);
	detector.getDescriptors(descriptors);


	std::vector<std::pair<float,int> > strengthIndex(keypoints.size());
	for(unsigned k = 0; k < keypoints.size(); k++)
	  {
	    strengthIndex[k].first = keypoints[k].response;
	    strengthIndex[k].second = k;
	  }

	save_keypoints(keypoints, descriptors, strengthIndex,inputFile, noDescriptor);
      }
    else
      {
	ASRL_ASSERT_GT(images.size(),0,"No image filenames passed to application");
	// A normal run...
	asrl::GpuSurfDetector detector(configuration);
	for(unsigned i = 0; i < images.size(); i++)
	  {
	    cv::Mat cvImage;
	    cvImage = cv::imread(images[i],CV_LOAD_IMAGE_GRAYSCALE);
	    
	    detector.buildIntegralImage(cvImage);

	    detector.detectKeypoints();
	    if(! noOrientation )
	      {
		if(fastOrientation)
		  detector.findOrientationFast();
		else
		  detector.findOrientation();
	      }

	    if(!noDescriptor)
	      {	
		detector.computeDescriptors();
	      }

	    std::vector<cv::KeyPoint> keypoints;
	    std::vector<float> descriptors;
	    detector.getKeypoints(keypoints);
	    detector.getDescriptors(descriptors);
	    

	    std::vector<std::pair<float,int> > strengthIndex(keypoints.size());
	    for(unsigned k = 0; k < keypoints.size(); k++)
	      {
		strengthIndex[k].first = keypoints[k].response;
		strengthIndex[k].second = k;
	      }

	    // Sort by strength but save the ordering.
	    if(targetFeatures > 0 && (int)keypoints.size() > targetFeatures)
	      {	
		std::partial_sort(strengthIndex.begin(), strengthIndex.begin() + targetFeatures, strengthIndex.end(), &interestingStrengthIndex);
		strengthIndex.resize(targetFeatures);
	      }

	    save_keypoints(keypoints, descriptors, strengthIndex, images[i], noDescriptor);

	    if(debugIimg)
	      {
		detector.saveIntegralImage(getFileBasename(images[i]));
	      }
	  }
      }

    
  }
  catch(std::exception const & e)
    {
      std::cout << "Exception during processing " << e.what() << std::endl;
    }

  return 0;
}

