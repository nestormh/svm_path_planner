/* 
 * File:   CRutaDB2.cpp
 * Author: neztol
 * 
 * Created on 30 de abril de 2010, 13:26
 */

#include "CRutaDB2.h"

CRutaDB2::CRutaDB2(const char * dbName, const char * staticRoute, const  char * rtRoute, const char * pathBase) {
    strcpy(dbStatic, staticRoute);
    strcpy(dbRT, rtRoute);
    strcpy(this->pathBase, pathBase);

    if (sqlite3_open(dbName, &db) != SQLITE_OK){
        cerr << "Error al abrir la base de datos: " << sqlite3_errmsg(db) << endl;
    }

    sqlite3_stmt *statement;

    const char *sql = "SELECT * FROM route where (name == ?);";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }

    sqlite3_bind_text(statement, 1, staticRoute, -1, NULL);
    if (sqlite3_step(statement) == SQLITE_ROW) {
        staticIndex = sqlite3_column_int(statement, 0);
    }
    if (sqlite3_reset(statement) != SQLITE_OK) {
        cerr << "Error al resetear el statement" << endl;
    }
    sqlite3_bind_text(statement, 1, rtRoute, -1, NULL);
    if (sqlite3_step(statement) == SQLITE_ROW) {
        realTimeIndex = sqlite3_column_int(statement, 0);
    }
    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }

    cout << staticIndex << ", " << realTimeIndex << endl;
    
    sql = "SELECT count(latitude) FROM points where (route == ?)";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }
    sqlite3_bind_int(statement, 1, staticIndex);
    if (sqlite3_step(statement) == SQLITE_ROW) {
        nStaticPoints = sqlite3_column_int(statement, 0);
    }
    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }

    cout << "nPoints = " << nStaticPoints << endl;
    
    sql = "SELECT count(latitude) FROM points where (route == ?)";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }
    sqlite3_bind_int(statement, 1, realTimeIndex);
    if (sqlite3_step(statement) == SQLITE_ROW) {
        nRTPoints = sqlite3_column_int(statement, 0);
    }
    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }

    cout << "nPoints = " << nRTPoints << endl;
    map = cvCreateImage(cvSize(480, 480), IPL_DEPTH_8U, 3);
    currentPoint = 0;
}

CRutaDB2::CRutaDB2(const CRutaDB2& orig) {
}

CRutaDB2::~CRutaDB2() {
    if (sqlite3_close(db) != SQLITE_OK) {
        cerr << "Error al cerrar la base de datos: " << sqlite3_errmsg(db) << endl;
    }

    cvReleaseImage(&map);
}

void CRutaDB2::getNextImage(IplImage * &imgRT, IplImage * &imgDB) {
    double localX, localY, angle;

    sqlite3_stmt *statement;
    char * sql = "SELECT localX, localY, angleIMU FROM points where (route == ?) AND (timestamp == ?);";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }
    sqlite3_bind_int(statement, 1, realTimeIndex);
    sqlite3_bind_int(statement, 2, currentPoint);

    if (sqlite3_step(statement) == SQLITE_ROW) {
        localX = sqlite3_column_double(statement, 0);
        localY = sqlite3_column_double(statement, 1);
        angle = sqlite3_column_double(statement, 2);
    }

    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }

    char imageName[1024];
    sprintf(imageName, "%s/%s/Camera2/Image%d.png", pathBase, dbRT, currentPoint);
    cout << imageName << endl;
    imgRT = cvLoadImage(imageName, 0);    
    imgDB = getNearestImage(localX, localY, angle);

    cvSetImageROI(imgRT, cvRect(5, 0, imgRT->width - 5, imgRT->height));
    cvSetImageROI(imgDB, cvRect(5, 0, imgDB->width - 5, imgDB->height));

    //cvNamedWindow("imgRT", 1);
    //cvShowImage("imgRT", imgRT);
    //cvNamedWindow("imgDB", 1);
    //cvShowImage("imgDB", imgDB);
    
    //drawAllPoints(staticPoints, rtPoints, true, true);*/
    currentPoint++;
    if (currentPoint == nRTPoints)
        currentPoint = 0;
    

}

IplImage * CRutaDB2::getNearestImage(double localX, double localY, double angle) {
    sqlite3_stmt *statement;
    cout << localX << ", " << localY << ", " << angle << endl;

    char * sql = "select timestamp, ((? - points.localX) * (? - points.localX) + (? - points.localY) * (? - points.localY)) as dist, ((abs(((360 + angleIMU)%360) - ?) + 360)%360) as difAng  from points where (((abs(((360 + angleIMU)%360) - ?) + 360)%360) < ?) and (route == ?) and (dist < ? * ?)";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }
    sqlite3_bind_double(statement, 1, localX);
    sqlite3_bind_double(statement, 2, localX);
    sqlite3_bind_double(statement, 3, localY);
    sqlite3_bind_double(statement, 4, localY);
    sqlite3_bind_double(statement, 5, angle);
    sqlite3_bind_double(statement, 6, angle);
    sqlite3_bind_double(statement, 7, ANGLE_THRESH);
    sqlite3_bind_int(statement, 8, staticIndex);
    sqlite3_bind_double(statement, 9, DIST_THRESH);
    sqlite3_bind_double(statement, 10, DIST_THRESH);

    double minDist = DBL_MAX;
    staticPoint = -1;

    while (sqlite3_step(statement) == SQLITE_ROW) {
        double tmpTimestamp = sqlite3_column_double(statement, 0);
        double tmpDist = sqlite3_column_int(statement, 2);
        if (minDist > tmpDist) {
            minDist = tmpDist;
            staticPoint = tmpTimestamp;
        }        
    }

    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }

    if (staticPoint == -1) {
        cerr << "No se encontraron vecinos para la imagen " << endl;
        IplImage * rtImg = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
        cvZero(rtImg);
        return rtImg;
    }

    char imageName[1024];
    IplImage * imgDB = NULL;
    sprintf(imageName, "%s/%s/Camera2/Image%d.png", pathBase, dbStatic, staticPoint);
    cout << imageName << endl;
    imgDB = cvLoadImage(imageName, 0);

    return imgDB;

}

void CRutaDB2::getImageAt(IplImage * &img, int type, int index) {
    char imageName[1024];
    if (type == 1)
        sprintf(imageName, "%s/%s/Camera2/Image%d.png", pathBase, dbRT, index);
    else sprintf(imageName, "%s/%s/Camera2/Image%d.png", pathBase, dbStatic, index);
    
    cout << imageName << endl;
    img = cvLoadImage(imageName, 0);

    cvSetImageROI(img, cvRect(5, 0, img->width - 5, img->height));    
}

void CRutaDB2::setCurrentPoint(int index) {
    currentPoint = index;
}

int CRutaDB2::getRTPoint() {
    return currentPoint;
}

int CRutaDB2::getSTPoint() {
    return staticPoint;
}


void CRutaDB2::getNextImage(IplImage * &imgRT, IplImage * &imgDB1, IplImage * &imgDB2, IplImage * &imgDB3) {
    double localX, localY, angle;

    sqlite3_stmt *statement;
    char * sql = "SELECT localX, localY, angleIMU FROM points where (route == ?) AND (timestamp == ?);";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }
    sqlite3_bind_int(statement, 1, realTimeIndex);
    sqlite3_bind_int(statement, 2, currentPoint);

    if (sqlite3_step(statement) == SQLITE_ROW) {
        localX = sqlite3_column_double(statement, 0);
        localY = sqlite3_column_double(statement, 1);
        angle = sqlite3_column_double(statement, 2);
    }

    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }

    char imageName[1024];
    sprintf(imageName, "%s/%s/Camera2/Image%d.png", pathBase, dbRT, currentPoint);
    cout << imageName << endl;
    imgRT = cvLoadImage(imageName, 0);
    getNearestImage(localX, localY, angle, imgDB1, imgDB2, imgDB3);

    cvSetImageROI(imgRT, cvRect(5, 0, imgRT->width - 5, imgRT->height));
    cvSetImageROI(imgDB1, cvRect(5, 0, imgDB1->width - 5, imgDB1->height));
    cvSetImageROI(imgDB2, cvRect(5, 0, imgDB1->width - 5, imgDB1->height));
    cvSetImageROI(imgDB3, cvRect(5, 0, imgDB1->width - 5, imgDB1->height));

    //cvNamedWindow("imgRT", 1);
    //cvShowImage("imgRT", imgRT);
    //cvNamedWindow("imgDB", 1);
    //cvShowImage("imgDB", imgDB);

    //drawAllPoints(staticPoints, rtPoints, true, true);*/
    currentPoint++;
    if (currentPoint == nRTPoints)
        currentPoint = 0;    
}

void CRutaDB2::getNearestImage(double localX, double localY, double angle, IplImage * &imgDB1, IplImage * &imgDB2, IplImage * &imgDB3) {
    sqlite3_stmt *statement;
    cout << localX << ", " << localY << ", " << angle << endl;

    char * sql = "select timestamp, ((? - points.localX) * (? - points.localX) + (? - points.localY) * (? - points.localY)) as dist, ((abs(((360 + angleIMU)%360) - ?) + 360)%360) as difAng  from points where (((abs(((360 + angleIMU)%360) - ?) + 360)%360) < ?) and (route == ?) and (dist < ? * ?)";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }
    sqlite3_bind_double(statement, 1, localX);
    sqlite3_bind_double(statement, 2, localX);
    sqlite3_bind_double(statement, 3, localY);
    sqlite3_bind_double(statement, 4, localY);
    sqlite3_bind_double(statement, 5, angle);
    sqlite3_bind_double(statement, 6, angle);
    sqlite3_bind_double(statement, 7, ANGLE_THRESH);
    sqlite3_bind_int(statement, 8, staticIndex);
    sqlite3_bind_double(statement, 9, DIST_THRESH);
    sqlite3_bind_double(statement, 10, DIST_THRESH);

    double minDist = DBL_MAX;
    staticPoint = -1;

    while (sqlite3_step(statement) == SQLITE_ROW) {
        double tmpTimestamp = sqlite3_column_double(statement, 0);
        double tmpDist = sqlite3_column_int(statement, 2);
        if (minDist > tmpDist) {
            minDist = tmpDist;
            staticPoint = tmpTimestamp;
        }
    }

    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }

    if (staticPoint == -1) {
        cerr << "No se encontraron vecinos para la imagen " << endl;
        imgDB1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
        imgDB2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
        imgDB3 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
        cvZero(imgDB1);
        cvZero(imgDB2);
        cvZero(imgDB3);

        return;
    }

    char imageName[1024];    
    sprintf(imageName, "%s/%s/Camera0/Image%d.png", pathBase, dbStatic, staticPoint);
    imgDB1 = cvLoadImage(imageName, 0);
    sprintf(imageName, "%s/%s/Camera1/Image%d.png", pathBase, dbStatic, staticPoint);
    imgDB2 = cvLoadImage(imageName, 0);
    sprintf(imageName, "%s/%s/Camera2/Image%d.png", pathBase, dbStatic, staticPoint);
    imgDB3 = cvLoadImage(imageName, 0);    
}
