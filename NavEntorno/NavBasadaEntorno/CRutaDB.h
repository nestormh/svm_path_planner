/* 
 * File:   CRutaDB.h
 * Author: neztol
 *
 * Created on 23 de noviembre de 2009, 9:47
 */

#ifndef _CRUTADB_H
#define	_CRUTADB_H

#include "ImageRegistration.h"
#include "GeographicLib/LocalCartesian.hpp"
#include "GeographicLib/Geocentric.hpp"
#include <sqlite3.h>

using namespace GeographicLib;
typedef Math::real real;

#define DB_NAME "/home/neztol/doctorado/Datos/DB/navEntorno.sqlite"
/*#define DB_STATIC "teguesteBaseDic08"
#define DB_RT "teguesteObsDic08"
#define DB_VERSION 2
#define INITIAL_POINT 255
#define PATH_BASE "/home/neztol/doctorado/Datos/DB/Rutas"//*/
/*#define DB_STATIC "IterNov08"
#define DB_RT "IterNov08Obst"
#define DB_VERSION 2
#define INITIAL_POINT 1
#define PATH_BASE "/home/neztol/doctorado/Datos/DB/Rutas" //*/
#define DB_STATIC "aerop14EneSinObs"
#define DB_RT "aerop14EneConObs"
#define DB_VERSION 1
#define INITIAL_POINT 1
#define PATH_BASE "/home/neztol/doctorado/Datos/Aerop" //*/

#define DIST_THRESH 1.0


class CRutaDB {
public:
    CRutaDB(const char * dbName, const char * staticRoute, const  char * rtRoute, const char * pathBase);
    CRutaDB(const CRutaDB& orig);
    virtual ~CRutaDB();

    void getNextImage(IplImage * &imgRT, IplImage * &imgDB);
    IplImage * getNearestImage(double latitude, double longitude, double height, double angle);

    int currentPoint;

private:
    sqlite3 * db;    
    int staticIndex;
    int realTimeIndex;
    LocalCartesian * lc;

    CvMat * rtPoints;
    CvMat * staticPoints;
    CvPoint2D32f local;

    char dbStatic[256];
    char dbRT[256];
    char pathBase[1024];

    IplImage * map;

    void drawAllPoints(CvMat * stPoints, CvMat * rtPoints, bool centered, bool angles);
};

void pruebaRutas();

#endif	/* _CRUTADB_H */

