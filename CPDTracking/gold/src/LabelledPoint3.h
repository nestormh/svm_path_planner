#ifndef _LABELLED_POINTS_H
#define _LABELLED_POINTS_H

#include <Data/Math/points.h>

template<class T> struct Point3;

template<class T>
struct LabelledPoint3 : public Point3 <T> {
    enum t_label { INF = 0, UNKNOWN = 1, GROUND = 2, VEHICLE = 3, INVALID = 4, VEHICLE_SIDE = 5 };
    
    t_label label;
        
    inline LabelledPoint3() {}
    inline LabelledPoint3(T _x, T _y, T _z, t_label _label = UNKNOWN) : Point3 <T> (_x, _y, _z), label(_label) {}

    template<class P>
    explicit LabelledPoint3(const P & _p, t_label _label = UNKNOWN) : Point3 <T> (_p), label(_label) {}
    template<class P>
    explicit LabelledPoint3(const Point3<P> & _p, t_label _label = UNKNOWN) : Point3 <T> (_p.x, _p.y, _p.z), label(_label) {}
    template<class P>
    LabelledPoint3(const LabelledPoint3<P> & _p) : Point3 <T> (_p.x, _p.y, _p.z), label(_p.label) {}    
};

/// Punto o Vettore (x,y,z,label) interi
typedef LabelledPoint3<int> LabelledPoint3i;
/// Punto o Vettore (x,y,z,label) precisione singola
typedef LabelledPoint3<float> LabelledPoint3f;
/// Punto o Vettore (x,y,z,label) precisione doppia
typedef LabelledPoint3<double> LabelledPoint3d;

#endif