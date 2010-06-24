/* 
 * File:   CImageSearch.h
 * Author: neztol
 *
 * Created on 24 de mayo de 2010, 11:50
 */

#ifndef _CIMAGESEARCH_H
#define	_CIMAGESEARCH_H

#include <sqlite3.h>
#include "stdafx.h"
#include "CRealMatches.h"
#include "fast/cvfast.h"

class CImageSearch {
public:
    CImageSearch(string dbName, string dbST, string dbRT, string pathBase, bool useIMU, CvRect rect, CvSize size);
    virtual ~CImageSearch();
    void getRTImage(IplImage * &imgRT);
    void getSTImage(IplImage * &imgST, int index = -1);

    void getInitialImage(IplImage * imgRT, IplImage * &imgST);
    void getNearestImage(IplImage * imgRT, IplImage * &imgST, int &code);
    void getNearestImage2(IplImage * imgRT, IplImage * &imgST, int &code);

    void startTest();

private:
    sqlite3 * db;

    string dbST;
    string dbRT;
    string pathBase;

    string extST;
    string extRT;

    int nRTPoints;
    int nStaticPoints;

    int indexST;
    int indexRT;

    CvMat * mapDistances;
    CvMat * mapAngles;
    CvMat * currentNearestPoints;

    IplImage * img1;
    IplImage * img2;
    
    vector <CvPoint2D32f> points1;
    vector <t_Pair> pairs;

    CvRect rect;
    CvSize size;

    void cleanRANSAC(int method, vector<t_Pair> &pairs);
    void testFast(IplImage * img, vector<CvPoint2D32f> &points);
    void findOFlowPairs(const IplImage * img1, const IplImage * img2, const CvPoint2D32f * origPoints, int nOrigFeat, CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &numberOfFeatures);
    void oFlow(vector <CvPoint2D32f> &points1, vector <t_Pair> &pairs, IplImage * &img1, IplImage * &img2);
    void checkCoveredArea(IplImage * imgB, IplImage * imgA, int &coveredArea);
    void setMaskFromPoints(IplImage * &mask, int index);
};

#endif	/* _CIMAGESEARCH_H */

