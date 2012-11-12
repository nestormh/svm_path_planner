#ifndef _STEREO_RECTIFICATION_TOOLS_H
#define _STEREO_RECTIFICATION_TOOLS_H

#include <Libs/Compatibility/DeclSpecs.h>
#include <Data/Base/CCameraParams.h>

/// Test if two cameras have the same intrinsic parameters
inline bool HaveCameraSameIntrinsic(const CameraParams & l, const CameraParams & r)
{
  return (l.ku == r.ku) &&
  (l.kv == r.kv) &&
  (l.u0 == r.u0) &&
  (l.v0 == r.v0);
}

/// Test if 2 cameras are aligned along Y axis and optical ray orthogonal to it.
inline bool AreCamerasStrictlyAligned(const CameraParams & l, const CameraParams & r)
{
return  (std::abs(l.roll)<0.0001) && (std::abs(r.roll)<0.0001) &&
	  (std::abs(l.yaw)<0.0001) && (std::abs(r.yaw)<0.0001) &&
	  (std::abs(l.pitch - r.pitch)<0.0001) && 
	  (std::abs(l.x - r.x)<0.001) &&
	  (std::abs(l.z - r.z)<0.001);
}

/// Test if 2 cameras are aligned along their epipole.
bool DECLSPEC_EXPORT AreCamerasAligned(const CameraParams & l, const CameraParams & r);

/// Elenco di algoritmi disponibili per ComputeRectificationParams
enum RectificationAlgo {
  LeftMaster,  ///< prende tutti i parametri della sx
  RightMaster, ///< prende i parametri della dx
  Average      ///< prende i valor medi tra le camere
};

/** Questa funzione permette di ottenere i parametri della camera rettificati partendo dai parametri originali della coppia stereo 
 * \code
 * ComputeRectificationParams(m_pLeftCamera->Params(), m_pRightCamera->Params(), m_leftRectifiedCameraParams, m_rightRectifiedCameraParams, RightMaster);
 * \endcode
 */
void DECLSPEC_EXPORT ComputeRectificationParams(const CameraParams & sl, const CameraParams & sr, CameraParams & dl, CameraParams & dr, RectificationAlgo algo);

/** Questa funzione permette di ottenere i parametri della camera rettificati partendo dai parametri originali della coppia stereo */
void DECLSPEC_EXPORT ComputeRectificationParams(const std::pair<CameraParams, CameraParams> & s, std::pair<CameraParams, CameraParams> &o, RectificationAlgo algo);

#endif
