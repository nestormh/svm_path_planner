#ifndef _DISPARITY_OPENCV_H
#define _DISPARITY_OPENCV_H

#include "DisparityBase.h"

#include <opencv2/opencv.hpp>

using namespace cimage;

template<typename T>
class TDisparityOpenCV {
    public:        
        TDisparityOpenCV(cv::StereoSGBM stereoSGBM, uint32_t width, uint32_t height) : 
                        TDisparityBase(width, height), m_stereoSGBM(stereoSGBM) {}
        
        // NOTE: Defaults are based on parameters given at 
        // http://www.cvlibs.net/datasets/kitti/eval_stereo_flow_detail.php?benchmark=stereo&error=3&eval=est&result=3ae300a3a3b3ed3e48a63ecb665dffcc127cf8ab
        TDisparityOpenCV(uint32_t width = 0, uint32_t height = 0, int minDisparity = 0, int numDisparities = 128, 
                         int SADWindowSize = 3, int P1 = 3 * 3 * 4, int P2 = 3 * 3 * 32, int disp12MaxDiff = 1, 
                         int preFilterCap = 63, int uniquenessRatio = 10, int speckleWindowSize = 100, 
                         int speckleRange = 32, bool fullDP = true);
                        
        virtual void calculate(const T & leftImage, const T & rightImage);                
        
    private:        
        cv::StereoSGBM m_stereoSGBM;
        CDSI m_dsi;
        
        uint32_t m_width;
        uint32_t m_height;
        
};

#endif