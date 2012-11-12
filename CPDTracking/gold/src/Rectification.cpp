#include "Rectification.h"

#include "StereoRectificationTools.h"

#include <Processing/Vision/PerspectiveMapping/affinemap.h>
#include <Processing/Vision/PerspectiveMapping/homography.h>

#include <utility>

#define __LOCATION__ __CLASS_METHOD__
// #define __LOCATION__ __FILE_LINE__
// #define VERBOSITY_DEBUG true
#include <Libs/Logger/Log.h>

std::pair<CameraParams, CameraParams> StereoRectification::Update(const dev::CCamera& left, const dev::CCamera& right, uint32_t width, uint32_t height, const std::string& left_lut_name, const std::string& right_lut_name, RectificationSteps steps, RectificationAlgo rectification_algo)
{
    std::pair<CameraParams, CameraParams> output_params;
    
    if(steps == RS_RECTIFY || steps == RS_UNDIST_RECTIFY)
        ComputeRectificationParams(std::make_pair(left.Params(), right.Params()), output_params, rectification_algo);
    else
    {
        output_params.first = left.Params();
        output_params.second = right.Params();
    }
    
    output_params.first.SetGeometry(width, height);
    output_params.second.SetGeometry(width, height);

    m_left_mono_rect.Update(left, output_params.first, left_lut_name, steps);
    m_right_mono_rect.Update(right, output_params.second, right_lut_name, steps);

    return output_params;
}

std::pair<CameraParams, CameraParams> StereoRectification::Update(const CameraParams& left, const CameraParams& right, uint32_t width, uint32_t height, const std::string& left_lut_filename, const std::string& right_lut_filename, RectificationSteps steps, RectificationAlgo rectification_algo)
{
    std::pair<CameraParams, CameraParams> output_params;

    if(steps == RS_RECTIFY || steps == RS_UNDIST_RECTIFY)
        ComputeRectificationParams(std::make_pair(left, right), output_params, rectification_algo);
    else
    {
        output_params.first = left;
        output_params.second = right;
    }

    output_params.first.SetGeometry(width, height);
    output_params.second.SetGeometry(width, height);

    m_left_mono_rect.Update(left, output_params.first, left_lut_filename, steps);
    m_right_mono_rect.Update(right, output_params.second, right_lut_filename, steps);

    return output_params;
}

CameraParams MonoRectification::Update(const dev::CCamera& camera, const CameraParams& dest_params, const std::string& lut_name, RectificationSteps steps)
{
    if(!camera.LUTs().empty())
    {
        dev::LUTMap::const_iterator ll = camera.LUTs().find(lut_name);
    
        return Update(camera.Params(), dest_params, (ll != camera.LUTs().end()) ? ll->second.native : lut_name, steps);
    }
    
    return Update(camera.Params(), dest_params, lut_name, steps);
}

CameraParams MonoRectification::Update(const CameraParams& src_params, const CameraParams& dest_params, const std::string& lut_filename, RectificationSteps steps)
{
    bool undist = (steps == RS_UNDIST) || (steps == RS_UNDIST_RECTIFY);
    bool rectify = (steps == RS_RECTIFY) || (steps == RS_UNDIST_RECTIFY);

    if(undist &&
            (!m_undist_lut ||
             m_undist_lut->SrcWidth() != src_params.width ||
             m_undist_lut->SrcHeight() != src_params.height))
    {
        m_undist_lut = boost::shared_ptr<BilinearImageLUT>(new BilinearImageLUT(src_params.width, src_params.height, src_params.width, src_params.height));
        try {
            log_info << "Loading LUT file: " << lut_filename << std::endl;
            if(!m_undist_lut->LoadFromFile(lut_filename.c_str())) throw("");
        } catch(...) {
            m_undist_lut.reset();
            log_error << "Invalid LUT file: " << lut_filename << std::endl;
            throw;
        }
    }

    if (!m_image_lut ||
            m_image_lut->GetWidth() != dest_params.width ||
            m_image_lut->GetHeight() != dest_params.height ||
            m_image_lut->SrcWidth() != src_params.width ||
            m_image_lut->SrcHeight() != src_params.height)
    {
        m_image_lut = boost::shared_ptr<BilinearImageLUT>(new BilinearImageLUT(dest_params.width, dest_params.height, src_params.width, src_params.height));
    }
    
    if(rectify)
    {
        if(undist && m_undist_lut && (m_src_params != src_params || m_dest_params != dest_params || m_steps != steps))
            m_image_lut->CreateFrom(ht::CameraRectify(src_params, dest_params), m_undist_lut.get());
        else if(m_src_params != src_params || m_dest_params != dest_params || m_steps != steps)
            m_image_lut->Compute(ht::CameraRectify(src_params, dest_params));
    }
    else if(undist && m_undist_lut && (m_src_params != src_params || m_dest_params != dest_params || m_steps != steps))
        m_image_lut->CreateFrom(at::resample(src_params.width, src_params.height, dest_params.width, dest_params.height), m_undist_lut.get());
    else if(m_src_params != src_params || m_dest_params != dest_params || m_steps != steps)
        m_image_lut->Compute(at::resample(src_params.width, src_params.height, dest_params.width, dest_params.height));

    m_src_params = src_params;
    m_dest_params = dest_params;
    m_steps = steps;

    return dest_params;
}
