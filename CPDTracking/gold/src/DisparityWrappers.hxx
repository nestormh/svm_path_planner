#include <opencv2/opencv.hpp>

#include <Data/CImage/Utilities.h>
// #include <Devices/Clock/CClock.h>
#include <Devices/Profiler/Profiler.h>

//NOTE: All methods receive left and right image, width, height, params related to the method, and eval map.
// At the end, the wrapper sets evals[KEY_OF_METHOD] = stuff
template <typename T>
void setDisparityOpenCV_SGM(const T & leftImage, const T & rightImage, const uint32_t & width, const uint32_t & height, CDeviceManager & devicemanager, const std::string & appName, 
                        t_OPENCV_SGM_params & params, const uint32_t & searchRange, std::map < t_evalType, t_disparityEvals > & evals) {
    
    if (params.P1 >= params.P2) {
        std::cerr << "P1 should be smaller than P2 in OpenCV SGM algorithm!!!!. Skipping..." << std::endl;
        return;
    }
    
    uint32_t searchRangeMax = searchRange;
    if (searchRange % 16 != 0) {
        searchRangeMax = (int)((double)searchRange / 16.0 + 0.5) * 16;
        std::cerr << "numDisparities = " << searchRange << ". It should be a multiple of 16. Using " << searchRangeMax << std::endl;
    }

    if (params.SADWindowSize % 2 == 0) {
        std::cerr << "SADWindowSize = " << params.SADWindowSize << ". It should be odd. Using " << params.SADWindowSize + 1 << std::endl;
        params.SADWindowSize = params.SADWindowSize + 1;
    }
    
    if (!evals[OPENCV_SGM].dsi) {
        evals[OPENCV_SGM].name = std::string("OPENCV_SGM");
        evals[OPENCV_SGM].initialPos = Point2d(-1, -1);
        evals[OPENCV_SGM].prevPos = Point2d(0, 0);
        evals[OPENCV_SGM].isRight = false;
        evals[OPENCV_SGM].dsi = cimage::Build<CDSI>(width, height);
        
        evals[OPENCV_SGM].dsi->Add("MIN", 0);                         
        evals[OPENCV_SGM].dsi->Add("MAX", searchRangeMax);   
        
        evals[OPENCV_SGM].timer = vltime::CChronometer("OPENCV_SGM_" + appName, vltime::CChronometer::REAL_TIME_CLOCK);
        
        evals[OPENCV_SGM].scale = 1.0;
        
        dev::CProfiler &  profiler  = static_cast<dev::CProfiler&>(devicemanager["Profiler"]);
        profiler.Connect(evals[OPENCV_SGM].timer);
    }
    
    cv::Mat left(cv::Size(width, height), CV_8UC1);
    cv::Mat right(cv::Size(width, height), CV_8UC1);    
    cv::Mat disp8(cv::Size(width, height), CV_64FC1);
    
    for (uint32_t y = 0; y < left.rows; y++) {
        for (uint32_t x = 0; x < left.cols; x++) {
            left.at<uchar>(y, x) = leftImage.Buffer() [y * width + x];
            right.at<uchar>(y, x) = rightImage.Buffer() [y * width + x];
        }
    }
        
    cv::StereoSGBM stereo(0, searchRangeMax, params.SADWindowSize,
                          params.SADWindowSize * params.SADWindowSize * params.P1, 
                          params.SADWindowSize * params.SADWindowSize * params.P2, 
                          params.disp12MaxDiff, params.preFilterCap, params.uniquenessRatio,
                          params.speckleWindowSize, params.speckleRange, params.fullDP);    
    
    cv::Mat disp;
    evals[OPENCV_SGM].timer.Start();
    stereo(left, right, disp);    
    evals[OPENCV_SGM].timer.Stop();
    
    disp.convertTo(disp8, CV_64F, 1./16.);
       
    // Results are stored into evals map    
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {            
            evals[OPENCV_SGM].dsi->Buffer()[y * width + x] = (double)disp8.at<double>(y, x);
            if (evals[OPENCV_SGM].dsi->Buffer()[y * width + x] == 0)
                evals[OPENCV_SGM].dsi->Buffer()[y * width + x] = DISPARITY_UNKNOWN;
        }
    }
}

template <typename T, typename E>
void setDisparityVislab(const t_evalType & evalType, const T & leftImage, const T & rightImage, const uint32_t & width, const uint32_t & height, CDeviceManager & devicemanager, const std::string & appName, 
                            E & disparityEngine, const double & searchRangeMax, std::map < t_evalType, t_disparityEvals > & evals) {
                                
    if (!evals[evalType].dsi) {
        switch (evalType) {
            case VISLAB_SGM: {
                evals[evalType].name = std::string("VISLAB_SGM");
                evals[evalType].timer = vltime::CChronometer("VISLAB_SGM_" + appName, vltime::CChronometer::REAL_TIME_CLOCK);
                evals[evalType].scale = 1.0;
                break;
            }
            case VISLAB_IFWTA: {
                evals[evalType].name = std::string("VISLAB_IFWTA");
                evals[evalType].timer = vltime::CChronometer("VISLAB_IFWTA_" + appName, vltime::CChronometer::REAL_TIME_CLOCK);
                evals[evalType].scale = 1.0;
                break;
            }
            case VISLAB_SGM_2: {
                evals[evalType].name = std::string("VISLAB_SGM_2");
                evals[evalType].timer = vltime::CChronometer("VISLAB_SGM_2_" + appName, vltime::CChronometer::REAL_TIME_CLOCK);
                evals[evalType].scale = 1.0;
                break;
            }
            case VISLAB_SGM_4: {
                evals[evalType].name = std::string("VISLAB_SGM_4");
                evals[evalType].timer = vltime::CChronometer("VISLAB_SGM_4_" + appName, vltime::CChronometer::REAL_TIME_CLOCK);
                evals[evalType].scale = 1.0;
                break;
            } 
            case VISLAB_ELAS: {
                evals[evalType].name = std::string("VISLAB_ELAS");
                evals[evalType].timer = vltime::CChronometer("VISLAB_ELAS_" + appName, vltime::CChronometer::REAL_TIME_CLOCK);
                evals[evalType].scale = 1.0;
                break;
            }
            default: {
                evals[evalType].name = std::string("UNKNOWN_ALGORITHM");
                evals[evalType].timer = vltime::CChronometer("UNKNOWN_ALGORITHM_" + appName, vltime::CChronometer::REAL_TIME_CLOCK);
                break;
            }
        }
        evals[evalType].initialPos = Point2d(-1, -1);
        evals[evalType].prevPos = Point2d(0, 0);
        evals[evalType].isRight = true;
        evals[evalType].dsi = cimage::Build<CDSI>(width, height);
        
        dev::CProfiler &  profiler  = static_cast<dev::CProfiler&>(devicemanager["Profiler"]);
        profiler.Connect(evals[evalType].timer);
    }

    std::vector<disparity::SearchRange> searchRanges(height, disparity::SearchRange(0, searchRangeMax));
    
    evals[evalType].timer.Start();
    boost::shared_ptr<const CDSI> dsi = disparityEngine.Run(leftImage, rightImage, searchRanges);
    evals[evalType].timer.Stop();
    
    *(evals[evalType].dsi) = (CDSI)*dsi;
    
}

