/* 
 * File:   CRoadDetection.h
 * Author: neztol
 *
 * Created on 13 de mayo de 2010, 13:03
 */

#ifndef _CROADDETECTION_H
#define	_CROADDETECTION_H

#include "CRutaDB2.h"
#include "ACO/CAntColony.h"

class CRoadDetection {
public:
    CRoadDetection(CvSize size);
    CRoadDetection(const CRoadDetection& orig);
    virtual ~CRoadDetection();
    void detect(int index, IplImage * result);
    void detectRoadWithFAST(int index, IplImage * result);
    void detectRoadWithFATPoints(int index, IplImage * result);
    void detectOcclusions(int index, IplImage * result);
    void detectACO(int index, IplImage * result);
    void detectACO(IplImage * img, IplImage * result);
    void detectFixed(IplImage * result);
private:
    CRutaDB2 * ruta;    
    IplImage * img1;
    IplImage * img2;
    IplImage * mask;
    IplImage * diff;
    IplImage * maskResult;

    CAntColony * aco;    

    CvSize size;

    void detectObstacles(IplImage * mask);
    void obstacleDetectionQuartile(IplImage * pcaResult, IplImage * mask);
    void calcPCA(IplImage * img1, IplImage * img2, IplImage * diff, IplImage * mask);

    void testFast(IplImage * img, vector<CvPoint2D32f> &points);
    void testShiTomasi(IplImage * img, vector<CvPoint2D32f> &points);
    void drawTriangles(vector<CvPoint2D32f> points, IplImage * input);
    void cleanNeighbors(vector<CvPoint2D32f> &points);
};

#endif	/* _CROADDETECTION_H */

