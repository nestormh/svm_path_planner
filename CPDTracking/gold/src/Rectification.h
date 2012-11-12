#ifndef _RECTIFICATION_H
#define _RECTIFICATION_H

/**********************************************************************
 *                                                                    *
 *  This file is part of the GOLD software                            *
 *                                                                    *
 *            University of Parma, Italy   1996-2011                  *
 *                                                                    *
 *       http://www.vislab.ce.unipr.it                                *
 *                                                                    *
 **********************************************************************/

/**
 * @file Rectification.h
 * @author Paolo Zani (zani@vislab.it)
 * @date 2011-10-17
 */  

#include "StereoRectificationTools.h"

#include <Data/CImage/TImage.h>
#include <Devices/Camera/CCamera.h>
#include <Processing/Vision/PerspectiveMapping/ImageLUT.h>

#include <boost/shared_ptr.hpp>

#include <string>

enum RectificationSteps
{
    RS_NONE = 0x0,
    RS_UNDIST = 0x2,
    RS_RECTIFY = 0x4,
    RS_UNDIST_RECTIFY = RS_UNDIST | RS_RECTIFY,
};

class DECLSPEC_EXPORT MonoRectification
{
    public:

        template<typename T>
        inline void Apply(const cimage::TImage<T>& source, cimage::TImage<T>& dest)
        {
            if(m_image_lut)
                m_image_lut->Apply(dest, source);
        }
        
        CameraParams Update(const dev::CCamera& camera, const CameraParams& dest_params, const std::string& lut_name = "", RectificationSteps steps = RS_UNDIST_RECTIFY);

        CameraParams Update(const CameraParams& src_params, const CameraParams& dest_params, const std::string& lut_filename = "", RectificationSteps steps = RS_UNDIST_RECTIFY);
        
    private:
        
        boost::shared_ptr<BilinearImageLUT> m_undist_lut;
        boost::shared_ptr<BilinearImageLUT> m_image_lut;
        CameraParams m_src_params;
        CameraParams m_dest_params;
        RectificationSteps m_steps;
};

class DECLSPEC_EXPORT StereoRectification
{
    public:
             
        template<typename T>
        inline void Apply(const cimage::TImage<T>& source_left, const cimage::TImage<T>& source_right, cimage::TImage<T>& dest_left, cimage::TImage<T>& dest_right)
        {
            m_left_mono_rect.Apply(source_left, dest_left);
            m_right_mono_rect.Apply(source_right, dest_right);
        }

        std::pair<CameraParams, CameraParams> Update(const dev::CCamera& left, const dev::CCamera& right, uint32_t width, uint32_t height, const std::string& left_lut_name, const std::string& right_lut_name, RectificationSteps steps = RS_UNDIST_RECTIFY, RectificationAlgo rectification_algo = RightMaster);

        std::pair<CameraParams, CameraParams> Update(const CameraParams& left, const CameraParams& right, uint32_t width, uint32_t height, const std::string& left_lut_filename, const std::string& right_lut_filename, RectificationSteps steps = RS_UNDIST_RECTIFY, RectificationAlgo rectification_algo = RightMaster);
        
    private:

        MonoRectification m_left_mono_rect;
        MonoRectification m_right_mono_rect;
};


#endif