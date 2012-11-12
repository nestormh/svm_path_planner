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
 * \file cpdApp.cpp
 * \author Néstor Morales Hernández (nestor@isaatc.ull.es)
 * \date 2012-10-11
 */

#include "cpdApp.h"

#include <Data/CImage/Utilities.h>
#include <Devices/Clock/CClock.h>
#include <Devices/Profiler/Profiler.h>
#include <Framework/CRecordingCtl.h>
#include <Framework/Transport.h>
#include <Framework/CSession.h>
#include <Processing/Vision/CImage/BasicOperations/BasicOperations.h>
#include <Processing/Vision/CImage/Conversions/CImageConversions.h>
#include <Processing/Vision/CImage/Filters/SobelFilter.h>
#include <Processing/Vision/CImage/Filters/GaussianFilter.h>
#include <Processing/SampleConsensus/RANSAC.h>
#include <Processing/SampleConsensus/Estimators/Plane.h>
#include <Processing/Vehicle/Transformation/VehicleTrajectory.h>

#include <Data/CINSData/CINSData.h>
#include <Data/CGPSData/CGPSData.h>

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include<stdio.h>

#define __LOCATION__ __CLASS_METHOD__
// #define __LOCATION__ __FILE_LINE__
// #define VERBOSITY_DEBUG true
#include <Libs/Logger/Log.h>

#include <boost/date_time/posix_time/posix_time.hpp>

using namespace cimage;
using namespace ui::conf;
using namespace ui::var;
using namespace ui::wgt;
using namespace ui::win;

void cpdApp::On_Initialization()
{
    m_synchro.SetApplication(*this);

    CDeviceNode& cameras = ( Dev() ["CAMERAS"] );
    CDeviceNode& ins = ( Dev() ["INS"] );
    CDeviceNode& gps = ( Dev() ["GPS"] );

    std::transform(cameras.Children().begin(), cameras.Children().end(), std::back_inserter(m_cameraNames), std::mem_fun(&CDeviceNode::Name));

    std::vector<std::string> INSNames;
    std::transform(ins.Children().begin(), ins.Children().end(), std::back_inserter(INSNames), std::mem_fun(&CDeviceNode::Name));

    std::vector<std::string> GPSNames;
    std::transform(gps.Children().begin(), gps.Children().end(), std::back_inserter(GPSNames), std::mem_fun(&CDeviceNode::Name));

    m_pRightCamera = Dev()["CAMERAS/" + INIFile()->Value("RIGHT CAMERA NAME", m_cameraNames[0])];
    m_pCenterCamera = Dev()["CAMERAS/" + INIFile()->Value("CENTER CAMERA NAME", m_cameraNames[0])];
    m_pLeftCamera = Dev()["CAMERAS/" + INIFile()->Value("LEFT CAMERA NAME", m_cameraNames[1])];
//     m_pLaserScanner = Dev()["LASERSCANNERS/" + INIFile()->Value("LASERSCANNER NAME", std::string("LUX"))];
//     m_pINS = Dev()["INS/" + INIFile()->Value("INS NAME", INSNames[0])];
//     m_pGPS = Dev()["GPS/" + INIFile()->Value("GPS NAME", GPSNames[0])];

    m_synchro.ConnectSync(*m_pLeftCamera);
    m_synchro.ConnectSync(*m_pCenterCamera);
    m_synchro.ConnectSync(*m_pRightCamera);
//     m_synchro.ConnectLast(*m_pLaserScanner);
//     m_synchro.ConnectBuffered<std::vector>(*m_pINS);
//     m_synchro.ConnectBuffered<std::vector>(*m_pGPS);

    Configuration conf(INIFile());

    Value<bool> useUndistLUT(&m_useUndistLUT);
    conf.Bind(useUndistLUT, "USE DEDISTORTION LUT", false);
    
    Value<bool> useRectificationLUT(&m_useRectificationLUT);
    conf.Bind(useRectificationLUT, "USE RECTIFICATION LUT", true);

    Value<std::string> leftWideLUTFilename(&m_leftWideLUTFilename);
    conf.Bind(leftWideLUTFilename, "LEFT WIDE LUT FILENAME", std::string());

    Value<std::string> leftShortLUTFilename(&m_leftShortLUTFilename);
    conf.Bind(leftShortLUTFilename, "LEFT SHORT LUT FILENAME", std::string());
   
    Value<std::string> rightWideLUTFilename(&m_rightWideLUTFilename);
    conf.Bind(rightWideLUTFilename, "RIGHT WIDE LUT FILENAME", std::string());
    
    Value<std::string> rightShortLUTFilename(&m_rightShortLUTFilename);
    conf.Bind(rightShortLUTFilename, "RIGHT SHORT LUT FILENAME", std::string());

    Value<std::string> pathBase(&m_pathBase);
    conf.Bind(pathBase, "PATH BASE", std::string("/tmp"));

    Value<uint32_t> width(&m_dstWidth);
    conf.Bind(width, "WIDTH", 0U);

    Value<uint32_t> height(&m_dstHeight);
    conf.Bind(height, "HEIGHT", 0U);

    Value<bool> showInputWindow(boost::bind<bool>(&CWindow::SetVisibility, &m_inputWindow, _1),
                                boost::bind<bool>(&CWindow::IsVisible, &m_inputWindow, false));
    conf.Bind(showInputWindow, "SHOW INPUT", false);

    panel.Label("cpdApp Main Panel").Geometry(600, 100)
    (
        VSizer()
        (
            TreeBook()
            (
                Page("Display")
                (
                    VSizer()
                    (
                        CheckBox(showInputWindow, "Show Input").Border(3)
                    )
                ),
                Page("Preprocessing")
                (
                    VSizer()
                    (
                        CheckBox(useUndistLUT, "Remove Distortion", (! m_leftWideLUTFilename.empty() &&
                                                                     ! m_leftShortLUTFilename.empty() &&
                                                                     ! m_rightWideLUTFilename.empty() &&
                                                                     ! m_rightShortLUTFilename.empty())).Border(3),
                        CheckBox(useRectificationLUT, "Rectify").Border(3)
                    )
                )
            )

        )
    );

    m_customResolution = (m_dstWidth && m_dstHeight) ? true : false;

    m_inputWindow.SetTitle("Input");

    if(m_customResolution) {
        m_inputWindow.SetSize(2 * m_dstWidth, 2 * m_dstHeight);
        m_inputWindow.SetVirtualView(2 * m_dstWidth, 2 * m_dstHeight);
    }

    // Initialization of OpenCV images
    leftOCV = cv::Mat(m_dstHeight, m_dstWidth, CV_8UC3);
    leftOCV = cv::Mat(m_dstHeight, m_dstWidth, CV_8UC3);
    centerOCV = cv::Mat(m_dstHeight, m_dstWidth, CV_8UC3);
    rightWideOCV = cv::Mat(m_dstHeight, m_dstWidth, CV_8UC3);
    rightShortOCV = cv::Mat(m_dstHeight, m_dstWidth, CV_8UC3);

    m_initTime = boost::posix_time::ptime(boost::posix_time::second_clock::universal_time());
    
    std::ostringstream ossFileNameList;
    ossFileNameList << m_pathBase << "/" << m_psession->IName() << "/" << m_psession->IName() << ".txt";
    std::ofstream fout(ossFileNameList.str().c_str(), std::ios::out | std::ios::trunc);
    fout << "LEFT " << m_pLeftCamera->Params().width << " " << m_pLeftCamera->Params().height << " " 
         << m_pLeftCamera->Params().ku << " " << m_pLeftCamera->Params().kv << " "
         << m_pLeftCamera->Params().u0 << " " << m_pLeftCamera->Params().v0 << " "
         << m_pLeftCamera->Params().x << " " << m_pLeftCamera->Params().y << " " << m_pLeftCamera->Params().z << " "
         << m_pLeftCamera->Params().yaw << " " << m_pLeftCamera->Params().pitch << " " << m_pLeftCamera->Params().roll << std::endl;
    fout << "CENTER " << m_pCenterCamera->Params().width << " " << m_pCenterCamera->Params().height << " " 
         << m_pCenterCamera->Params().ku << " " << m_pCenterCamera->Params().kv << " "
         << m_pCenterCamera->Params().u0 << " " << m_pCenterCamera->Params().v0 << " "
         << m_pCenterCamera->Params().x << " " << m_pCenterCamera->Params().y << " " << m_pCenterCamera->Params().z << " "
         << m_pCenterCamera->Params().yaw << " " << m_pCenterCamera->Params().pitch << " " << m_pCenterCamera->Params().roll << std::endl;
    fout << "RIGHT " << m_pRightCamera->Params().width << " " << m_pRightCamera->Params().height << " " 
         << m_pRightCamera->Params().ku << " " << m_pRightCamera->Params().kv << " "
         << m_pRightCamera->Params().u0 << " " << m_pRightCamera->Params().v0 << " "
         << m_pRightCamera->Params().x << " " << m_pRightCamera->Params().y << " " << m_pRightCamera->Params().z << " "
         << m_pRightCamera->Params().yaw << " " << m_pRightCamera->Params().pitch << " " << m_pRightCamera->Params().roll << std::endl;
    fout.close();
}

void cpdApp::On_Initialization(Transport& tr)
{
    m_synchro.SetTransport(tr);
}

void cpdApp::On_ShutDown()
{
}

void cpdApp::On_Session_Open(const CSession& session)
{
    m_psession=&session;
}


void cpdApp::On_Execute()
{

    // TODO: Revisar si quiero sincronizar un poco mejor los frames
    CImage::SharedPtrConstType left, center, right;

//     std::vector<dev::CINS::FrameType> insFrames;
//     std::vector<dev::CGPS::FrameType> gpsFrames;

    {
        boost::mutex::scoped_lock lock(m_synchro.Mutex());
        m_synchro.Wait(lock);

        left = m_synchro.SyncFrameFrom<dev::CCamera>(*m_pLeftCamera).Data;
        center = m_synchro.SyncFrameFrom<dev::CCamera>(*m_pCenterCamera).Data;
        right = m_synchro.SyncFrameFrom<dev::CCamera>(*m_pRightCamera).Data;
//         m_laserScan = m_synchro.LastFrameFrom(*m_pLaserScanner).Data;
//         m_synchro.BufferedFrameFrom<std::vector>(*m_pINS).swap(insFrames);
//         m_synchro.BufferedFrameFrom<std::vector>(*m_pGPS).swap(gpsFrames);

        log_info << " Processing frames: " << m_synchro.SyncFrameFrom<dev::CCamera>(*m_pLeftCamera).TimeStamp << ", " << m_synchro.SyncFrameFrom<dev::CCamera>(*m_pRightCamera).TimeStamp << std::endl;
    }

    m_srcWidth = left->W();
    m_srcHeight = left->H();

    if(! m_customResolution) {

        if(m_dstWidth != m_srcWidth || m_dstHeight != m_srcHeight) {

            m_inputWindow.SetSize(2 * m_srcWidth, 2 * m_srcHeight);
            m_inputWindow.SetVirtualView(2 * m_srcWidth, 2 * m_srcHeight);
        }

        m_dstWidth = m_srcWidth;
        m_dstHeight = m_srcHeight;
    }

    m_leftRGB = Build<CImageRGB8>(m_dstWidth, m_dstHeight);    
    m_centerRGB = Build<CImageRGB8>(m_dstWidth, m_dstHeight);
    m_rightShortRGB = Build<CImageRGB8>(m_dstWidth, m_dstHeight);
    m_rightWideRGB = Build<CImageRGB8>(m_dstWidth, m_dstHeight);
    m_leftTmpRGB = Build<CImageRGB8>(m_srcWidth, m_srcHeight);
    m_centerTmpRGB = Build<CImageRGB8>(m_srcWidth, m_srcHeight);
    m_rightTmpRGB = Build<CImageRGB8>(m_srcWidth, m_srcHeight);
    
    Convert(*left, *m_leftTmpRGB, BAYER_DECODING_LUMINANCE);
    Convert(*center, *m_centerTmpRGB, BAYER_DECODING_LUMINANCE);
    Convert(*right, *m_rightTmpRGB, BAYER_DECODING_LUMINANCE);

    
    // TODO: Procesar laserScan
    /*m_worldPointsLaser.clear();
    m_worldPointsLaser.reserve(m_laserScan.Data.size());
    for (uint32_t i = 0; i < m_laserScan.Data.size(); i++) {
        m_worldPointsLaser.push_back(m_laserScan.Data[i]);
    }*/

    // TODO: Procesar GPS (Ver si lo puedo obtener en coordenadas UTM)
//     if (gpsFrames.size() != 0)
//         if (gpsFrames.back().Data.capabilities.test(CGPSData::CAP_SPEED))
//             m_trajectoryStep = gpsFrames.back().Data.speed;


    // TODO: Procesar IMU
//     if (insFrames.size() != 0)
//         if (insFrames.back().Data.capabilities.test(CINSData::CAP_YAW_RATE))
//             if (m_trajectoryStep != 0)
//                 m_curvature = insFrames.back().Data.yaw_rate / m_trajectoryStep;
//             else m_curvature = 0;


    pubishImages();

    Output();

    m_synchro.EndOfProcessing();
}

void cpdApp::Output()
{
    if(m_inputWindow.IsVisible()) {

        m_inputWindow.Clear();
        m_inputWindow.DrawImage(0, 0, m_dstWidth, m_dstHeight, m_leftRGB);
        m_inputWindow.DrawImage(m_dstWidth, 0, m_dstWidth, m_dstHeight, m_rightWideRGB);
        m_inputWindow.DrawImage(0, m_dstHeight, m_dstWidth, m_dstHeight, m_centerRGB);
        m_inputWindow.DrawImage(m_dstWidth, m_dstHeight, m_dstWidth, m_dstHeight, m_rightShortRGB);        
        m_inputWindow.Refresh();
    }
}

void cpdApp::pubishImages() {
        
    RectificationSteps steps = RS_NONE;

    if(m_useUndistLUT && m_useRectificationLUT)
        steps = RS_UNDIST_RECTIFY;
    else if(m_useUndistLUT)
        steps = RS_UNDIST;
    else if(m_useRectificationLUT)
        steps = RS_RECTIFY;

    m_lCameraWorldParams = m_pLeftCamera->Params();
    m_cCameraWorldParams = m_pCenterCamera->Params();
    m_rCameraWorldParams = m_pRightCamera->Params();

    m_rCameraWorldParams = m_pRightCamera->Params();    
    m_rCameraWorldParams.ku = m_pCenterCamera->Params().ku;
    m_rCameraWorldParams.kv = m_pCenterCamera->Params().kv;
    m_rCameraWorldParams.u0 = m_pCenterCamera->Params().u0;
    m_rCameraWorldParams.v0 = m_pCenterCamera->Params().v0;
    
    std::pair<CameraParams, CameraParams> m_rectified;
    
    m_rectified = m_rectification.Update(m_cCameraWorldParams, m_rCameraWorldParams, 
                                         m_dstWidth, m_dstHeight, 
                                         m_leftShortLUTFilename, m_rightShortLUTFilename, steps);
    m_cCameraWorldParams = m_rectified.first;
    m_rCameraWorldParams = m_rectified.second;
    
    m_rectification.Apply(*m_centerTmpRGB, *m_rightTmpRGB, *m_centerRGB, *m_rightShortRGB);
         
    m_rCameraWorldParams = m_pRightCamera->Params();    
    m_rCameraWorldParams.ku = m_pLeftCamera->Params().ku;
    m_rCameraWorldParams.kv = m_pLeftCamera->Params().kv;
    m_rCameraWorldParams.u0 = m_pLeftCamera->Params().u0;
    m_rCameraWorldParams.v0 = m_pLeftCamera->Params().v0;
    
    m_rectified = m_rectification.Update(m_lCameraWorldParams, m_rCameraWorldParams, 
                                         m_dstWidth, m_dstHeight, 
                                         m_leftWideLUTFilename, m_rightWideLUTFilename, steps);
    m_lCameraWorldParams = m_rectified.first;
    m_rCameraWorldParams = m_rectified.second;
    
    m_rectification.Apply(*m_leftTmpRGB, *m_rightTmpRGB, *m_leftRGB, *m_rightWideRGB);
    
    m_lCameraWorldParams.SetGeometry(m_dstWidth, m_dstHeight);
    m_cCameraWorldParams.SetGeometry(m_dstWidth, m_dstHeight);
    m_rCameraWorldParams.SetGeometry(m_dstWidth, m_dstHeight);        
        
    for (int y = 0; y < leftOCV.rows; y++) {
        for (int x = 0; x < leftOCV.cols; x++) {
            leftOCV.ptr<uchar>(y)[3 * x] = m_leftRGB->Buffer()[y * m_dstWidth + x].B;
            leftOCV.ptr<uchar>(y)[3 * x + 1] = m_leftRGB->Buffer()[y * m_dstWidth + x].G;
            leftOCV.ptr<uchar>(y)[3 * x + 2] = m_leftRGB->Buffer()[y * m_dstWidth + x].R;

            centerOCV.ptr<uchar>(y)[3 * x] = m_centerRGB->Buffer()[y * m_dstWidth + x].B;
            centerOCV.ptr<uchar>(y)[3 * x + 1] = m_centerRGB->Buffer()[y * m_dstWidth + x].G;
            centerOCV.ptr<uchar>(y)[3 * x + 2] = m_centerRGB->Buffer()[y * m_dstWidth + x].R;

            rightWideOCV.ptr<uchar>(y)[3 * x] = m_rightWideRGB->Buffer()[y * m_dstWidth + x].B;
            rightWideOCV.ptr<uchar>(y)[3 * x + 1] = m_rightWideRGB->Buffer()[y * m_dstWidth + x].G;
            rightWideOCV.ptr<uchar>(y)[3 * x + 2] = m_rightWideRGB->Buffer()[y * m_dstWidth + x].R;

            rightShortOCV.ptr<uchar>(y)[3 * x] = m_rightShortRGB->Buffer()[y * m_dstWidth + x].B;
            rightShortOCV.ptr<uchar>(y)[3 * x + 1] = m_rightShortRGB->Buffer()[y * m_dstWidth + x].G;
            rightShortOCV.ptr<uchar>(y)[3 * x + 2] = m_rightShortRGB->Buffer()[y * m_dstWidth + x].R;
        }
    }

    
    std::ostringstream ossFileNameBase;
    ossFileNameBase << m_pathBase << "/" << m_psession->IName();
    
    struct stat st;
    if (stat(ossFileNameBase.str().c_str(), &st) != 0)
        if((mkdir(ossFileNameBase.str().c_str(), 00777))==-1) {
            std::cerr << "Error in creating dir: " << ossFileNameBase.str() << std::endl;
        }
        
    std::ostringstream ossFileNameLW;
    ossFileNameLW << ossFileNameBase.str() << "/image" << std::setfill('0') << std::setw(7) << m_synchro.SyncFrameFrom<dev::CCamera>(*m_pLeftCamera).Counter << "LW.bmp";
    cv::imwrite(ossFileNameLW.str(), leftOCV);

    std::ostringstream ossFileNameLS;
    ossFileNameLS << ossFileNameBase.str() << "/image" << std::setfill('0') << std::setw(7) << m_synchro.SyncFrameFrom<dev::CCamera>(*m_pLeftCamera).Counter << "LS.bmp";
    cv::imwrite(ossFileNameLS.str(), centerOCV);
    
    std::ostringstream ossFileNameRW;
    ossFileNameRW << ossFileNameBase.str() << "/image" << std::setfill('0') << std::setw(7) << m_synchro.SyncFrameFrom<dev::CCamera>(*m_pLeftCamera).Counter << "RW.bmp";
    cv::imwrite(ossFileNameRW.str(), rightWideOCV);
    
    std::ostringstream ossFileNameRS;
    ossFileNameRS << ossFileNameBase.str() << "/image" << std::setfill('0') << std::setw(7) << m_synchro.SyncFrameFrom<dev::CCamera>(*m_pLeftCamera).Counter << "RS.bmp";
    cv::imwrite(ossFileNameRS.str(), rightShortOCV);
    
    std::ostringstream ossFileNameList;
    ossFileNameList << ossFileNameBase.str() << "/" << m_psession->IName() << ".txt";
    std::ofstream fout(ossFileNameList.str().c_str(), std::ios::out | std::ios::app);
    fout << m_synchro.SyncFrameFrom<dev::CCamera>(*m_pLeftCamera).TimeStamp << "\t" 
         << "image" << std::setfill('0') << std::setw(7) << m_synchro.SyncFrameFrom<dev::CCamera>(*m_pLeftCamera).Counter << "LW.bmp" << "\t"
         << "image" << std::setfill('0') << std::setw(7) << m_synchro.SyncFrameFrom<dev::CCamera>(*m_pLeftCamera).Counter << "LS.bmp" << "\t"
         << "image" << std::setfill('0') << std::setw(7) << m_synchro.SyncFrameFrom<dev::CCamera>(*m_pLeftCamera).Counter << "RW.bmp" << "\t"
         << "image" << std::setfill('0') << std::setw(7) << m_synchro.SyncFrameFrom<dev::CCamera>(*m_pLeftCamera).Counter << "RS.bmp" << std::endl;
    fout.close();
}

REGISTER_APPLICATION(cpdApp, "CPD");

#include <Engine/CommandersRegistration.h>

REGISTER_UI ( CmdrCPD, "CPDCommander" );