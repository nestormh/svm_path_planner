#ifndef _DISPARITY_H
#define _DISPARITY_H

/*
    Copyright (c) 2012, Néstor Morales Hernández <nestor@isaatc.ull.es>
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

    THIS SOFTWARE IS PROVIDED BY Néstor Morales Hernández <email> ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Néstor Morales Hernández <email> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * \file cpdApp.h
 * \author Néstor Morales Hernández (nestor@isaatc.ull.es)
 * \date 2012-10-11
 */

#include "Application.h"
#include "Rectification.h"
#include "DisparityDefs.h"

#include <Devices/Camera/CCamera.h>
#include <Devices/LaserScanner/CLaserScanner.h>
#include <Data/CImage/Images/CImageMono8.h>
#include <Data/CImage/Images/CImageMono8s.h>
#include <Data/CImage/Images/CImageRGB8.h>
#include <Framework/Synchronizers.h>
#include <Processing/Vision/Stereo/Images/CDSI.h>
#include <Processing/Vision/Stereo/DisparityEngine/UI_DisparityEngine.h>
#include <Processing/Vision/PerspectiveMapping/ImageLUT.h>
#include <UI/CWindows/CWindow.h>
#include <UI/Panel/Panel.h>
#include <Data/Math/Plane.h>

#include <deque>

#include <Engine/Commands.h>
#include <Engine/Commander.h>
#include <Engine/CEngineInterface.h>

#include <boost/shared_ptr.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <Devices/INS/CINS.h>
#include <Devices/GPS/CGPS.h>

#include <opencv2/opencv.hpp>

#include <stdint.h>
#include <string>
#include <vector>

CEngineInterface* g_pEngine = NULL;

struct CmdrCPD: public Commander {
    virtual void On_Initialization() {
        g_pEngine = &Engine();
    }
};

class cpdApp : public CApplication {
public:

    DECL_METHODS;

    virtual void On_Initialization();

    virtual void On_Initialization(Transport& tr);

    virtual void On_ShutDown();

    virtual void On_Execute();

    virtual void On_Session_Open(const CSession& session);

private:
    void pubishImages();

    void Output();

    const CSession* m_psession;

    bool m_customResolution;

    uint32_t m_srcWidth;
    uint32_t m_srcHeight;

    uint32_t m_dstWidth;
    uint32_t m_dstHeight;

    std::string m_leftWideLUTFilename;
    std::string m_leftShortLUTFilename;
    std::string m_rightWideLUTFilename;
    std::string m_rightShortLUTFilename;

    std::vector<std::string> m_cameraNames;
    dev::CCamera* m_pLeftCamera;
    dev::CCamera* m_pCenterCamera;
    dev::CCamera* m_pRightCamera;
    dev::CLaserScanner* m_pLaserScanner;
    dev::CINS* m_pINS;
    dev::CGPS* m_pGPS;

    CameraParams m_rCameraWorldParams;
    CameraParams m_cCameraWorldParams;
    CameraParams m_lCameraWorldParams;

    boost::shared_ptr<cimage::CImageRGB8> m_leftTmpRGB;
    boost::shared_ptr<cimage::CImageRGB8> m_centerTmpRGB;
    boost::shared_ptr<cimage::CImageRGB8> m_rightTmpRGB;
    boost::shared_ptr<cimage::CImageRGB8> m_leftRGB;
    boost::shared_ptr<cimage::CImageRGB8> m_centerRGB;
    boost::shared_ptr<cimage::CImageRGB8> m_rightShortRGB;
    boost::shared_ptr<cimage::CImageRGB8> m_rightWideRGB;

    cv::Mat leftOCV, centerOCV, rightWideOCV, rightShortOCV;

    StereoRectification m_rectification;

    CScan m_laserScan;

    bool m_useUndistLUT;
    bool m_useRectificationLUT;

    ui::win::CWindow m_inputWindow;

    Synchronizer_Basic m_synchro;

    boost::posix_time::ptime m_initTime;
    
    std::string m_pathBase;
};

#endif
