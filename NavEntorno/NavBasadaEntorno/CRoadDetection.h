/* 
 * File:   CRoadDetection.h
 * Author: neztol
 *
 * Created on 13 de mayo de 2010, 13:03
 */

#ifndef _CROADDETECTION_H
#define	_CROADDETECTION_H

#include "CRutaDB2.h"

class CRoadDetection {
public:
    CRoadDetection(CvSize size);
    CRoadDetection(const CRoadDetection& orig);
    virtual ~CRoadDetection();
    void detect(int index, IplImage * result);
private:
    CRutaDB2 * ruta;    
    IplImage * img1;
    IplImage * img2;
    IplImage * mask;
    IplImage * diff;
    IplImage * maskResult;

    CvSize size;

    void detectObstacles(IplImage * mask);
    void obstacleDetectionQuartile(IplImage * pcaResult, IplImage * mask);
    void calcPCA(IplImage * img1, IplImage * img2, IplImage * diff, IplImage * mask);
};

#endif	/* _CROADDETECTION_H */

