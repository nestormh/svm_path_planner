/* 
 * File:   CImageNavigation.h
 * Author: neztol
 *
 * Created on 21 de junio de 2010, 16:39
 */

#ifndef _CIMAGENAVIGATION_H
#define	_CIMAGENAVIGATION_H

#include "ViewMorphing.h"
#include "ImageRegistration.h"
#include "ACO/CAntColony.h"
#include "CRutaDB2.h"
#include <sqlite3.h>
#include <map>
#include "Surf/imload.h"
#include "Surf/surflib.h"
#include "Surf/os_mapping.h"
#include "CRealMatches.h"
#include "fast/cvfast.h"

class CImageNavigation {
public:
    CImageNavigation(string route);
    CImageNavigation(const CImageNavigation& orig);
    virtual ~CImageNavigation();

    void makePairs();
    void makePairsOFlow();
private:
    vector< Ipoint > findSURF(Image *im, double thresh, int &VLength);
    void testSurf(IplImage * img1, IplImage * img2);
    double distSquare(double *v1, double *v2, int n);
    int findMatch(const Ipoint& ip1, const vector< Ipoint >& ipts, int vlen);
    vector< int > findMatches(const vector< Ipoint >& ipts1, const vector< Ipoint >& ipts2, int vlen);
    void cleanRANSAC(int method, vector<t_Pair> &pairs);

    void findOFlowPairs(const IplImage * img1, const IplImage * img2, const CvPoint2D32f * origPoints, int nOrigFeat, CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &numberOfFeatures);
    void oFlow(vector <CvPoint2D32f> &points1, vector <t_Pair> &pairs, IplImage * &img1, IplImage * &img2);
    void testFast(IplImage * img, vector<CvPoint2D32f> &points);

    CvSize size;
    vector <t_Pair> pairs;
    vector <string> fileNames;

    string route;
};

#endif	/* _CIMAGENAVIGATION_H */

