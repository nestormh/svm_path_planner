#include "StereoRectificationTools.h"

#include <cmath>
#include <Processing/Math/Transformation.h>
#include <Data/Math/points.h>
#include <Data/Math/TMatrix.h>

template<class T>
inline Point3<T> operator % (const Point3<T> &a, const Point3<T> & b)
{
   return Point3<T>(a.y * b.z - a.z * b.y, -(a.x * b.z - a.z * b.x), a.x * b.y - a.y * b.x);
}

bool AreCamerasAligned(const CameraParams & l, const CameraParams & r)
{
  if(std::abs(l.yaw - r.yaw) < 0.0001 &&
    std::abs(l.pitch - r.pitch) < 0.0001 &&
    std::abs(l.roll - r.roll) < 0.0001)
  {
  Point3d v = Point3d(l.x - r.x, l.y - r.y, l.z - r.z);
  Point3d y = Column<1>(Rotation(l.yaw, l.pitch, l.roll));
  return std::abs(v * y) > 0.999; // dovrebbe essere 1 o -1
  }

return false;
}

void MatrixToTaitBryanAngles(double & yaw, double & pitch, double & roll, const TMatrix<double, 3,3> & r)
{
  // 012
  // 345
  // 678
  roll = atan2(r[7], r[8]);
  // r[7]/r[6] = tan(yaw)
  yaw = atan2(r[3], r[0]);
  // r[6]*r[6] + r[7]*r[7] = cos(pitch)^2
  // r[2]*r[2] + r[5]*r[5] = cos(pitch)^2
  // -r[8]/sqrt(cos(pitch)^2) = tan(pitch)
  pitch = atan2(-r[6], sqrt(r[0]*r[0] + r[3]*r[3]));
}

void ComputeRectificationParams(const CameraParams & sl, const CameraParams & sr, CameraParams & dl, CameraParams & dr, RectificationAlgo algo)
{
  dl.x = sl.x;
  dl.y = sl.y;
  dl.z = sl.z;

  dr.x = sr.x;
  dr.y = sr.y;
  dr.z = sr.z;
  
  // Intrinseci
    switch(algo)
    {
      case LeftMaster:
	dr.ku = dl.ku = sl.ku;
	dr.kv = dl.kv = sl.kv;
	dr.u0 = dl.u0 = sl.u0;
	dr.v0 = dl.v0 = sl.v0;
	dl.width = dr.width = sl.width;
	dl.height = dr.height = sl.height;
      break;
      case RightMaster:
	dr.ku = dl.ku = sr.ku;
	dr.kv = dl.kv = sr.kv;
	dr.u0 = dl.u0 = sr.u0;
	dr.v0 = dl.v0 = sr.v0;
	dl.width = dr.width = sr.width;
	dl.height = dr.height = sr.height;
      break;
      case Average:
	dr.ku = dl.ku = (sl.ku + sr.ku) * 0.5;
	dr.kv = dl.kv = (sl.kv + sr.kv) * 0.5;
	dr.u0 = dl.u0 = (sl.u0 + sr.u0) * 0.5;
	dr.v0 = dl.v0 = (sl.v0 + sr.v0) * 0.5;
	dl.width = dr.width = (sr.width + sl.width) * 0.5;
	dl.height = dr.height = (sr.height + sl.height) * 0.5;
      break;
    }
      
  
  if( (std::abs(sl.x - sr.x) < 0.001) && (std::abs(sl.z - sr.z) < 0.001) )
  {
    // CASO 1: camere allineate lungo l'asse Y
    dl.yaw = dr.yaw = 0.0;
    dl.roll = dr.yaw = 0.0;
    
    switch(algo)
    {
      case LeftMaster:
	dr.pitch = dl.pitch = sl.pitch;
      break;
      case RightMaster:
	dr.pitch = dl.pitch = sr.pitch;
      break;
      case Average:
	dr.pitch = dl.pitch = (sr.pitch + sl.pitch) * 0.5;
      break;
    }
    
  }
  else
  {
  
  // CASO 2: camere arbitrariamente posizionate

  // l'asse X e' dominante in quanto e' nella direzione dell'immagine
  // L'asse Y e' quello che congiunge i pin-hole
  // Z e' perpendicolare ai 2
  
  // vettore che congiunge la camera sx alla dx (y)
  Point3d v = Point3d(sl.x - sr.x, sl.y - sr.y, sl.z - sr.z);
  v.normalize();
  
    switch(algo)
    {
      case LeftMaster:
      {
	// richiedo la matrice per convertire da coordinate sensore a coordinate mondo.
	// le colonne di R sono la direzione degli assi
	TMatrix<double, 3,3> R = Rotation(sl.yaw, sl.pitch, sl.roll);
	Point3d x = Column<0>(R); // vettore x
	Point3d z = x % v; // vettore z
	Point3d x1 = v % z; // vettore x'
	z.normalize();
	x1.normalize();
	
	Column<0>(R, x1); Column<1>(R, v); Column<2>(R, z);

	MatrixToTaitBryanAngles(dr.yaw, dr.pitch, dr.roll, R);
	dl.yaw = dr.yaw;
	dl.pitch = dr.pitch;
	dl.roll = dr.roll;
      }
      break;
	
      case RightMaster:
      {
	// richiedo la matrice per convertire da coordinate sensore a coordinate mondo.
	// le colonne di R sono la direzione degli assi
	TMatrix<double, 3,3> R = Rotation(sr.yaw, sr.pitch, sr.roll);
    
	Point3d x = Column<0>(R); // vettore x
	
	Point3d z = x % v; // vettore z
	
	Point3d x1 = v % z; // vettore x'
	
	z.normalize();
	x1.normalize();
	
	Column<0>(R, x1); Column<1>(R, v); Column<2>(R, z);
    
	MatrixToTaitBryanAngles(dr.yaw, dr.pitch, dr.roll, R);

	dl.yaw = dr.yaw;
	dl.pitch = dr.pitch;
	dl.roll = dr.roll;
      }
      break;
      
      case Average:
      {
	TMatrix<double, 3,3> Rl = Rotation(sl.yaw, sl.pitch, sl.roll);
	TMatrix<double, 3,3> Rr = Rotation(sr.yaw, sr.pitch, sr.roll);
	
	Point3d xl = Column<0>(Rl); // vettore xleft
	Point3d xr = Column<0>(Rr); // vettore xright
	
	Point3d x = xl + xr;
// 	x.normalize();
	Point3d z = x % v; // vettore z
	Point3d x1 = v % z; // vettore x'
	z.normalize();
	x1.normalize();
	
	TMatrix<double, 3,3> R;
	Column<0>(R, x1); Column<1>(R, v); Column<2>(R, z);
	
	MatrixToTaitBryanAngles(dr.yaw, dr.pitch, dr.roll, R);
	dl.yaw = dr.yaw;
	dl.pitch = dr.pitch;
	dl.roll = dr.roll;
      }
      break;      
	      
    }
  
  }
}

void ComputeRectificationParams(const std::pair<CameraParams, CameraParams> & s, std::pair<CameraParams, CameraParams> &o, RectificationAlgo algo)
{
  ComputeRectificationParams(s.first, s.second, o.first, o.second, algo);
}