#ifndef _DISPARITY_DEFS_H
#define _DISPARITY_DEFS_H

#include <Processing/Vision/Stereo/Images/CDSI.h>
#include <Data/CImage/Images/CImageMono8.h>
#include <Processing/Vision/Stereo/DisparityEngine/UI_DisparityEngine.h>

#include <string>

enum t_evalType { VISLAB_SGM = 0, OPENCV_SGM = 1, VISLAB_IFWTA = 2, VISLAB_SGM_2 = 3, VISLAB_SGM_4 = 4, VISLAB_ELAS = 5 };

enum t_VizType { REPROJECTED = 0, DIFFERENCE = 1, STEREO = 2, FILTER = 3 };

template<typename ResultType_Agg, uint32_t Threads_Agg, typename ResultType_Opt, typename Impl_Opt, uint32_t Threads_Opt>
const bool disparity::opt::FilteredWTA<ResultType_Agg, Threads_Agg, ResultType_Opt, Impl_Opt, Threads_Opt>::DEFAULT_DISAMB_ENABLED =false;

template<typename ResultType_Agg, uint32_t Threads_Agg, typename ResultType_Opt, typename Impl_Opt, uint32_t Threads_Opt>
const bool disparity::opt::FilteredWTA<ResultType_Agg, Threads_Agg, ResultType_Opt, Impl_Opt, Threads_Opt>::DEFAULT_SUBPIXEL_ACCURACY_ENABLED = true;

template<typename ResultType_Agg, uint32_t Threads_Agg, typename ResultType_Opt, typename Impl_Opt, uint32_t Threads_Opt>
const bool disparity::opt::FilteredWTA<ResultType_Agg, Threads_Agg, ResultType_Opt, Impl_Opt, Threads_Opt>::DEFAULT_EXTRA_FILTERING_ENABLED = false;

template<typename ResultType_Agg, uint32_t Threads_Agg, typename ResultType_Opt, typename Impl_Opt, uint32_t Threads_Opt>
const bool disparity::opt::FilteredWTA<ResultType_Agg, Threads_Agg, ResultType_Opt, Impl_Opt, Threads_Opt>::DEFAULT_FILTERING_ENABLED = false;

template<typename ResultType_Agg, uint32_t Threads_Agg, typename ResultType_Opt, typename Impl_Opt, uint32_t Threads_Opt>
const int32_t disparity::opt::FilteredWTA<ResultType_Agg, Threads_Agg, ResultType_Opt, Impl_Opt, Threads_Opt>::DEFAULT_MIN_CORRELATION_FUNCTION_ELEMENTS_NUM = 16;

template<typename ResultType_Agg, uint32_t Threads_Agg, typename ResultType_Opt, typename Impl_Opt, uint32_t Threads_Opt>
const double disparity::opt::FilteredWTA<ResultType_Agg, Threads_Agg, ResultType_Opt, Impl_Opt, Threads_Opt>::DEFAULT_MAX_CORRELATION_PEAK_DISPERSION = 30.0;

template<typename ResultType_Agg, uint32_t Threads_Agg, typename ResultType_Opt, typename Impl_Opt, uint32_t Threads_Opt>
const double disparity::opt::FilteredWTA<ResultType_Agg, Threads_Agg, ResultType_Opt, Impl_Opt, Threads_Opt>::DEFAULT_MIN_INV_CORRELATION_PEAK_SHARPNESS = 10.0;

template<typename ResultType_Agg, uint32_t Threads_Agg, typename ResultType_Opt, typename Impl_Opt, uint32_t Threads_Opt>
const double disparity::opt::FilteredWTA<ResultType_Agg, Threads_Agg, ResultType_Opt, Impl_Opt, Threads_Opt>::DEFAULT_MAX_CORRELATION_PEAK_VALUE = 10.0;

typedef struct {
  
    boost::shared_ptr<CDSI> dsi;    
    
    std::string name;
    
    bool isRight;
    
    vltime::CChronometer timer;
    
    // NCC
    float ncc;
    
    // False correspondences
    double fc;
    double fc_INF;
    
    // Leader vehicle measures
    double lp;
    double w;
    double distDiff;
    
    // Seed points
    Point2d initialPos;
    Point2d prevPos;
    Point3d initialPos3d;        
    
    // Downsample scale
    double scale;
    
    // Use Sobel or not
    bool useSobel;
    
    // Use Gaussian Filter or not
    bool useGaussian;
    
    // Max/min corners for LV
    Point3d minCorner;
    Point3d maxCorner;
    Point3d minCornerLaser;
    Point3d maxCornerLaser;
    
    // Percentage of points found by the algorithm
    double retrievedPoints;
    
} t_disparityEvals;

typedef struct {
    bool useSobel;
} t_disparityParams;


typedef struct {    
    int SADWindowSize; 
    int P1; 
    int P2; 
    int disp12MaxDiff; 
    int preFilterCap; 
    int uniquenessRatio;
    
    int speckleWindowSize; 
    int speckleRange; 
    bool fullDP;
} t_OPENCV_SGM_params;

ui::var::Range<uint32_t> m_nSamplesVar;

template <typename T>
void setDisparityOpenCV_SGM(const T & leftImage, const T & rightImage, const uint32_t & width, const uint32_t & height, CDeviceManager & devicemanager, const std::string & appName, 
                            t_OPENCV_SGM_params & params, const uint32_t & searchRange, std::map < t_evalType, t_disparityEvals > & evals);

template <typename T, typename E>
void setDisparityVislab(const t_evalType & evalType, const T & leftImage, const T & rightImage, const uint32_t & width, const uint32_t & height, CDeviceManager & devicemanager, const std::string & appName, 
                            E & disparityEngine, const double & searchRangeMax, std::map < t_evalType, t_disparityEvals > & evals);

inline uint32_t getIndexScaled(const Point2d & point, const uint32_t & width, const double & scale) {
    return (uint32_t)(point.y / scale) * (uint32_t)(width / scale) + (uint32_t)(point.x / scale);
}

#include "DisparityWrappers.hxx"

#endif