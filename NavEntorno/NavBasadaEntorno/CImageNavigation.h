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
#include "CRealMatches.h"
#include "fast/cvfast.h"

class CImageNavigation {
public:
    CImageNavigation(string route, string ext);
    CImageNavigation(const CImageNavigation& orig);
    virtual ~CImageNavigation();
    
    void makePairsOFlow();
private:
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

