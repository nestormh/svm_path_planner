
#ifndef _llstestunit_h
#define _llstestunit_h

#include "ap.h"

#include "spline3.h"
#include "reflections.h"
#include "lq.h"
#include "bidiagonal.h"
#include "rotations.h"
#include "bdsvd.h"
#include "qr.h"
#include "blas.h"
#include "svd.h"
#include "leastsquares.h"


bool testlls(bool silent);


bool isglssolution(int n,
     int m,
     const ap::real_1d_array& y,
     const ap::real_1d_array& w,
     const ap::real_2d_array& fmatrix,
     const ap::real_1d_array& c);


/*************************************************************************
Silent unit test
*************************************************************************/
bool llstestunit_test_silent();


/*************************************************************************
Unit test
*************************************************************************/
bool llstestunit_test();


#endif
