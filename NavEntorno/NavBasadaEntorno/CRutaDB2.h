/* 
 * File:   CRutaDB2.h
 * Author: neztol
 *
 * Created on 30 de abril de 2010, 13:26
 */

#ifndef _CRUTADB2_H
#define	_CRUTADB2_H

#include "ImageRegistration.h"
#include <sqlite3.h>

#define ANGLE_THRESH 45
#define DIST_THRESH 1.5

#define TYPE_RT 1
#define TYPE_ST 2

class CRutaDB2 {
public:
    CRutaDB2(const char * dbName, const char * staticRoute, const  char * rtRoute, const char * pathBase, CvRect roi, CvSize size);
    CRutaDB2(const CRutaDB2& orig);
    virtual ~CRutaDB2();

    void getNextImage(IplImage * &imgRT, IplImage * &imgDB);
    IplImage * getNearestImage(double localX, double localY, double angle);
    void getNextImage(IplImage * &imgRT, IplImage * &imgDB1, IplImage * &imgDB2, IplImage * &imgDB3);
    void getNearestImage(double localX, double localY, double angle, IplImage * &imgDB1, IplImage * &imgDB2, IplImage * &imgDB3);
    void setCurrentPoint(int index);
    void getImageAt(IplImage * &img, int type, int index);

    int getRTPoint();
    int getSTPoint();

    int getMaxSTPoint();
    int getMaxRTPoint();

private:
    sqlite3 * db;
    int staticIndex;
    int realTimeIndex;

    CvPoint2D32f local;

    int nRTPoints;
    int nStaticPoints;

    char dbStatic[256];
    char dbRT[256];
    char pathBase[1024];

    int staticPoint;
    int currentPoint;

    IplImage * map;

    CvSize size;
    CvRect roi;

    void drawAllPoints(CvMat * stPoints, CvMat * rtPoints, bool centered, bool angles);
};

#endif	/* _CRUTADB2_H */

