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

class CRutaDB2 {
public:
    CRutaDB2(const char * dbName, const char * staticRoute, const  char * rtRoute, const char * pathBase);
    CRutaDB2(const CRutaDB2& orig);
    virtual ~CRutaDB2();

    void getNextImage(IplImage * &imgRT, IplImage * &imgDB);
    IplImage * getNearestImage(double localX, double localY, double angle);
    void setCurrentPoint(int index);
    void getImageAt(IplImage * &img, int type, int index);

    int currentPoint;

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

    IplImage * map;

    void drawAllPoints(CvMat * stPoints, CvMat * rtPoints, bool centered, bool angles);
};

#endif	/* _CRUTADB2_H */

