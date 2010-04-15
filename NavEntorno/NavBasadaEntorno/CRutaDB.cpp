/* 
 * File:   CRutaDB.cpp
 * Author: neztol
 * 
 * Created on 23 de noviembre de 2009, 9:47
 */

#include "CRutaDB.h"

CRutaDB::CRutaDB(const char * dbName, const char * staticRoute, const  char * rtRoute, const char * pathBase) {    
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

    int nStaticPoints = -1;
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
    staticPoints = cvCreateMat(1, nStaticPoints, CV_32FC3);
    char ** staticDates = new char*[nStaticPoints];

    sql = "SELECT latitude, longitude, msl, date FROM points where (route == ?)";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }

    int i = 0;
    double lastX = DBL_MIN, lastY = DBL_MIN;
    double * stAngle = new double[nStaticPoints];
    sqlite3_bind_int(statement, 1, staticIndex);
    while (sqlite3_step(statement) == SQLITE_ROW) {
        Math::real latitude = sqlite3_column_double(statement, 0);
        Math::real longitude = sqlite3_column_double(statement, 1);
        Math::real height = sqlite3_column_double(statement, 2);
        
        staticDates[i] = new char[23];
        strcpy(staticDates[i], (char *)sqlite3_column_text(statement, 3));

        if (i == 0) {
            try {
                lc = new LocalCartesian(latitude, longitude, height);
            } catch (int e) {
                cout << "Error inicializando conversor de coordenadas. Excepción número " << e << endl;
            }
        }

        Math::real x, y, z;
        lc->Forward(latitude, longitude, height, x, y, z);
        cvSet2D(staticPoints, 0, i, cvScalar(x, y, z));        

        if ((lastX == DBL_MIN) || (lastY == DBL_MIN)) {
            stAngle[i] = 0;
        } else {
            stAngle[i] = atan2(y - lastY, x - lastX);
            if (stAngle[i] < 0) {
                stAngle[i] += CV_PI * 2;
            }            
        }
        lastX = x;
        lastY = y;

        i++;
    }
    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }

    int nRTPoints = -1;
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
    rtPoints = cvCreateMat(1, nRTPoints, CV_32FC3);
    char ** rtDates = new char*[nRTPoints];
    
    sql = "SELECT latitude, longitude, msl, date FROM points where (route == ?)";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }

    i = 0;
    lastX = lastY = DBL_MIN;
    double * rtAngle = new double[nRTPoints];
    sqlite3_bind_int(statement, 1, realTimeIndex);
    while (sqlite3_step(statement) == SQLITE_ROW) {
        Math::real latitude = sqlite3_column_double(statement, 0);
        Math::real longitude = sqlite3_column_double(statement, 1);
        Math::real height = sqlite3_column_double(statement, 2);
        rtDates[i] = new char[23];
        strcpy(rtDates[i], (char *)sqlite3_column_text(statement, 3));

        if (i == 0) {
            try {
                lc = new LocalCartesian(latitude, longitude, height);
            } catch (int e) {
                cout << "Error inicializando conversor de coordenadas. Excepción número " << e << endl;
            }
        }

        Math::real x, y, z;
        lc->Forward(latitude, longitude, height, x, y, z);
        cvSet2D(rtPoints, 0, i, cvScalar(x, y, z));

        if ((lastX == DBL_MIN) || (lastY == DBL_MIN)) {
            rtAngle[i] = 0;
        } else {
            rtAngle[i] = atan2(y - lastY, x - lastX);
            if (rtAngle[i] < 0) {
                rtAngle[i] += CV_PI * 2;
            }
        }
        lastX = x;
        lastY = y;

        i++;
    }
    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }

    /*sql = "UPDATE points SET localX = ?, localY = ?, localZ = ?, angleGPS = ? WHERE (route == ?) and (date == ?)";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }
    for (i = 0; i < nStaticPoints; i++) {
        CvScalar pt = cvGet2D(staticPoints, 0, i);        
        sqlite3_bind_double(statement, 1, pt.val[0]);
        sqlite3_bind_double(statement, 2, pt.val[1]);
        sqlite3_bind_double(statement, 3, pt.val[2]);
        sqlite3_bind_double(statement, 4, stAngle[i]);
        sqlite3_bind_int(statement, 5, staticIndex);
        sqlite3_bind_text(statement, 6, staticDates[i], 23, NULL);
        
        if (sqlite3_step(statement) != SQLITE_DONE) {
            cerr << "Error al ejecutar el statement" << endl;
        }

        if (sqlite3_reset(statement) != SQLITE_OK) {
            cerr << "Error al finalizar el statement" << endl;
        }
    }
    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }

    sql = "UPDATE points SET localX = ?, localY = ?, localZ = ?, angleGPS = ? WHERE (route == ?) and (date == ?)";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }
    for (i = 0; i < nRTPoints; i++) {
        CvScalar pt = cvGet2D(rtPoints, 0, i);
        sqlite3_bind_double(statement, 1, pt.val[0]);
        sqlite3_bind_double(statement, 2, pt.val[1]);
        sqlite3_bind_double(statement, 3, pt.val[2]);
        sqlite3_bind_double(statement, 4, rtAngle[i]);
        sqlite3_bind_int(statement, 5, realTimeIndex);
        sqlite3_bind_text(statement, 6, rtDates[i], 23, NULL);

        if (sqlite3_step(statement) != SQLITE_DONE) {
            cerr << "Error al ejecutar el statement" << endl;
        }

        if (sqlite3_reset(statement) != SQLITE_OK) {
            cerr << "Error al finalizar el statement" << endl;
        }
    }
    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }//*/

    map = cvCreateImage(cvSize(480, 480), IPL_DEPTH_8U, 3);
    currentPoint = INITIAL_POINT;
}

CRutaDB::CRutaDB(const CRutaDB& orig) {
}

CRutaDB::~CRutaDB() {
    if (sqlite3_close(db) != SQLITE_OK) {
        cerr << "Error al cerrar la base de datos: " << sqlite3_errmsg(db) << endl;
    }

    cvReleaseImage(&map);
    cvReleaseMat(&staticPoints);
    cvReleaseMat(&rtPoints);
}

void CRutaDB::drawAllPoints(CvMat * stPoints, CvMat * rtPoints, bool centered, bool angles) {
    map->origin = 1;
    cvZero(map);

    CvMat * pts1 = cvCreateMat(1, stPoints->cols, CV_32FC2);
    CvMat * pts2 = cvCreateMat(1, rtPoints->cols, CV_32FC2);

    CvPoint2D32f ul = cvPoint2D32f(DBL_MAX, DBL_MAX);
    CvPoint2D32f lr = cvPoint2D32f(DBL_MIN, DBL_MIN);    

    for (int i = 0; i < stPoints->cols; i++) {    
        CvScalar pt = cvGet2D(stPoints, 0, i);

        cvSet2D(pts1, 0, i, pt);

        if ((pt.val[0] < -10000) || (pt.val[0] > 10000)) continue;
        if ((pt.val[1] < -10000) || (pt.val[1] > 10000)) continue;
        if (ul.x > pt.val[0]) ul.x = pt.val[0];
        if (ul.y > pt.val[1]) ul.y = pt.val[1];
        if (lr.x <= pt.val[0]) lr.x = pt.val[0];
        if (lr.y <= pt.val[1]) lr.y = pt.val[1];
    }
    for (int i = 0; i < rtPoints->cols; i++) {
        CvScalar pt = cvGet2D(rtPoints, 0, i);

        cvSet2D(pts2, 0, i, pt);

        if ((pt.val[0] < -10000) || (pt.val[0] > 10000)) continue;
        if ((pt.val[1] < -10000) || (pt.val[1] > 10000)) continue;
        if (ul.x > pt.val[0]) ul.x = pt.val[0];
        if (ul.y > pt.val[1]) ul.y = pt.val[1];
        if (lr.x <= pt.val[0]) lr.x = pt.val[0];
        if (lr.y <= pt.val[1]) lr.y = pt.val[1];
    }

    if (centered) {
        CvScalar pt = cvGet2D(rtPoints, 0, currentPoint);
        ul.x = pt.val[0] - 3;
        lr.x = pt.val[0] + 3;
        ul.y = pt.val[1] - 3;
        lr.y = pt.val[1] + 3;
    } else {
        double distX = abs(ul.x - lr.x);
        double distY = abs(ul.y - lr.y);

        ul.x -= 0.1 * distX;
        ul.y -= 0.1 * distY;
        lr.x += 0.1 * distX;
        lr.y += 0.1 * distY;
    }
    
    CvPoint2D32f src[4] = { cvPoint2D32f(ul.x, ul.y), cvPoint2D32f(lr.x, ul.y), cvPoint2D32f(lr.x, lr.y), cvPoint2D32f(ul.x, lr.y) };
    CvPoint2D32f dst[4] = { cvPoint2D32f(0, 0), cvPoint2D32f(map->width, 0), cvPoint2D32f(map->width, map->height), cvPoint2D32f(0, map->height) };
    CvMat * transf = cvCreateMat(2, 3, CV_32FC1);
    cvGetAffineTransform(src, dst, transf);

    cvTransform(pts1, pts1, transf);
    cvTransform(pts2, pts2, transf);

    CvMat * selection = cvCreateMat(1, 1, CV_32FC2);
    cvSet2D(selection, 0, 0, cvScalar(local.x, local.y, 0));
    cvTransform(selection, selection, transf);

    CvScalar scPt = cvGet2D(selection, 0, 0);
    CvPoint myPt = cvPoint(scPt.val[0], scPt.val[1]);

    double lastX = DBL_MIN, lastY = DBL_MIN;
    double angle = 0;
    for (int i = 0; i < stPoints->cols; i++) {
        CvScalar pt = cvGet2D(pts1, 0, i);
        
        if ((lastX == DBL_MIN) || (lastY == DBL_MIN)) {
            angle = 0;            
        } else {
            angle = atan2(pt.val[1] - lastY, pt.val[0] - lastX);
            if (angle < 0) angle += CV_PI * 2;
        }        

        double distX = 20 * cos(angle);
        double distY = 20 * sin(angle);

        if (angles) cvLine(map, cvPoint(pt.val[0], pt.val[1]), cvPoint(pt.val[0] + distX, pt.val[1] + distY), cvScalar(255, 0, 0));
        //cvLine(map, cvPoint(pt.val[0], pt.val[1]), cvPoint(lastX, lastY), cvScalar(0, 255, 0));
        cvCircle(map, myPt, 2, cvScalar(255, 255, 0), 3);
        cvCircle(map, cvPoint(pt.val[0], pt.val[1]), 2, cvScalar(255, 0, 0), -1);

        lastX = pt.val[0];
        lastY = pt.val[1];

    }
    lastX = DBL_MIN, lastY = DBL_MIN;
    for (int i = 0; i < rtPoints->cols; i++) {
        CvScalar pt = cvGet2D(pts2, 0, i);
        
        if ((lastX == DBL_MIN) || (lastY == DBL_MIN)) {
            angle = 0;
        } else {
            angle = atan2(pt.val[1] - lastY, pt.val[0] - lastX);
            if (angle < 0) angle += CV_PI;
        }

        double distX = 20 * cos(angle);
        double distY = 20 * sin(angle);

        //cvLine(map, cvPoint(pt.val[0], pt.val[1]), cvPoint(pt.val[0] + distX, pt.val[1] + distY), cvScalar(0, 0, 255));
        cvLine(map, cvPoint(pt.val[0], pt.val[1]), cvPoint(lastX, lastY), cvScalar(128, 128, 128));
        cvCircle(map, cvPoint(pt.val[0], pt.val[1]), 2, cvScalar(0, 0, 255), -1);
        if (i == currentPoint) {
            cvCircle(map, cvPoint(pt.val[0], pt.val[1]), 2, cvScalar(255, 0, 255), -1, CV_AA);
            cvCircle(map, cvPoint(pt.val[0], pt.val[1]), 80, cvScalar(255, 0, 255), 1, CV_AA);
            cvCircle(map, cvPoint(pt.val[0], pt.val[1]), 240, cvScalar(255, 0, 255), 1, 4 | CV_AA);
            if (angles) cvLine(map, cvPoint(pt.val[0], pt.val[1]), cvPoint(pt.val[0] + distX, pt.val[1] + distY), cvScalar(255, 0, 255));
        }

        lastX = pt.val[0];
        lastY = pt.val[1];

    }

    cvNamedWindow("AllPoints", 1);
    cvShowImage("AllPoints", map);        
}

void CRutaDB::getNextImage(IplImage * &imgRT, IplImage * &imgDB) {    
    double latitude, longitude, height, angle;
    
    sqlite3_stmt *statement;
    char * sql = "SELECT latitude, longitude, height, angleGPS FROM points where (route == ?)";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }
    sqlite3_bind_int(statement, 1, realTimeIndex);
    for (int i = 0; i <= currentPoint; i++) {
        if (sqlite3_step(statement) != SQLITE_ROW) {
            cerr << "Se llegó al final de la ruta" << endl;
            if (cvWaitKey(5000) == 27)
                exit(0);
            currentPoint = 0;
            getNextImage(imgRT, imgDB);
            return;
        }
    }
    latitude = sqlite3_column_double(statement, 0);
    longitude = sqlite3_column_double(statement, 1);
    height = sqlite3_column_double(statement, 2);
    angle = sqlite3_column_double(statement, 3);

    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }

    char imageName[1024];
    if (DB_VERSION == 1) {
        sprintf(imageName, "%s/%s/Imagen%da.jpg", pathBase, dbRT, currentPoint);
        imgRT = cvLoadImage(imageName, 0);
        cvNamedWindow("imgRT", 1);
        cvShowImage("imgRT", imgRT);
        imgDB = getNearestImage(latitude, longitude, height, angle);
        cvNamedWindow("imgDB", 1);
        cvShowImage("imgDB", imgDB);
    } else {
        sprintf(imageName, "%s/%s/Imagen%d_0.jpg", pathBase, dbRT, currentPoint);
        imgRT = cvLoadImage(imageName, 0);
        cvNamedWindow("imgRT", 1);
        cvShowImage("imgRT", imgRT);
        imgDB = getNearestImage(latitude, longitude, height, angle);
        cvNamedWindow("imgDB", 1);
        cvShowImage("imgDB", imgDB);
    }

    drawAllPoints(staticPoints, rtPoints, true, true);
    currentPoint++;
    
}

IplImage * CRutaDB::getNearestImage(double latitude, double longitude, double height, double angle) {
    double localX, localY, localZ;
    lc->Forward(latitude, longitude, height, localX, localY, localZ);
    sqlite3_stmt *statement;

    char * sql = "SELECT localX, localY, angleGPS, date FROM points where (route = ?) and ((? - points.localX) * (? - points.localX) + (? - points.localY) * (? - points.localY) < ? * ?)";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }
    sqlite3_bind_int(statement, 1, staticIndex);
    sqlite3_bind_double(statement, 2, localX);
    sqlite3_bind_double(statement, 3, localX);
    sqlite3_bind_double(statement, 4, localY);
    sqlite3_bind_double(statement, 5, localY);
    sqlite3_bind_double(statement, 6, DIST_THRESH);
    sqlite3_bind_double(statement, 7, DIST_THRESH);
    double minDistAng = 3 * CV_PI;
    local = cvPoint2D32f(DBL_MIN, DBL_MIN);
    char date[24];
    int nRadio = 0;
    while (sqlite3_step(statement) == SQLITE_ROW) {        
        double tmpX = sqlite3_column_double(statement, 0);
        double tmpY = sqlite3_column_double(statement, 1);
        double tmpAngle = sqlite3_column_double(statement, 2);
        char * tmpDate = (char *)sqlite3_column_text(statement, 3);        

        double minAngle = min(angle, tmpAngle);
        double maxAngle = max(angle, tmpAngle);

        double distAng = maxAngle - minAngle;
        if (distAng > CV_PI) {
            minAngle += CV_PI * 2;
            distAng = minAngle - maxAngle;
        }

        if (distAng < minDistAng) {
            minDistAng = distAng;
            local = cvPoint2D32f(tmpX, tmpY);
            strcpy(date, tmpDate);
        }

        nRadio++;        
    }

    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }

    if ((nRadio == 0) || (minDistAng > CV_PI / 4)) {
        cerr << "No se encontraron vecinos para la imagen " << endl;
        //if (cvWaitKey(0) == 27) exit(0);
        return NULL;
    }

    sql = "SELECT date, latitude, longitude, height FROM points where (route = ?)";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }
    sqlite3_bind_int(statement, 1, staticIndex);    

    int nIndex = 0;
    double localLat = 0, localLong = 0, localAlt = 0;
    while (sqlite3_step(statement) == SQLITE_ROW) {
        char * tmpDate = (char *)sqlite3_column_text(statement, 0);
        localLat = sqlite3_column_double(statement, 1);
        localLong = sqlite3_column_double(statement, 2);
        localAlt = sqlite3_column_double(statement, 3);
        
        if (strcmp(date, tmpDate) == 0) break;        

        nIndex++;
    }

    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }

    char imageName[1024];
    IplImage * imgDB = NULL;
    if (DB_VERSION == 1) {
        sprintf(imageName, "%s/%s/Imagen%da.jpg", pathBase, dbStatic, nIndex);
        imgDB = cvLoadImage(imageName, 0);                
    } else {
        sprintf(imageName, "%s/%s/Imagen%d_0.jpg", pathBase, dbStatic, nIndex);
        imgDB = cvLoadImage(imageName, 0);
    }

    cout << latitude << ", " << longitude << ", " << height << " >>> ";
    cout << localLat << ", " << localLong << ", " << localAlt << endl;

    return imgDB;
}

void pruebaRutas() {
    cvDestroyWindow("Img1");
    cvDestroyWindow("Img2");
    CRutaDB dbRuta(DB_NAME, DB_STATIC, DB_RT, PATH_BASE);
    CImageRegistration registration(cvSize(320, 240));
    IplImage * imgRT, * imgDB, * imgRT2, * imgDB2;
    imgRT2 = cvCreateImage(cvSize(310, 240), IPL_DEPTH_8U, 1);
    imgDB2 = cvCreateImage(cvSize(310, 240), IPL_DEPTH_8U, 1);
    while (true) {
        dbRuta.getNextImage(imgRT, imgDB);
        if (imgDB == NULL) continue;
        /*cvSetImageROI(imgRT, cvRect(0, 0, 310, 240));
        cvSetImageROI(imgDB, cvRect(0, 0, 310, 240));
        cvCopyImage(imgRT, imgRT2);
        cvCopyImage(imgDB, imgDB2);
        cvResetImageROI(imgRT);
        cvResetImageROI(imgDB);
        cvResize(imgRT2, imgRT, CV_INTER_LINEAR);
        cvResize(imgDB2, imgDB, CV_INTER_LINEAR);*/
        //registration.registration(imgDB, NULL, imgRT);
        cout << dbRuta.currentPoint << endl;
        if (cvWaitKey(0) == 27) {
            exit(0);
        }
        cvReleaseImage(&imgRT);
        cvReleaseImage(&imgDB);
    }
}

