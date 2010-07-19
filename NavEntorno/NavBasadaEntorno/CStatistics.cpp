/* 
 * File:   CStatistics.cpp
 * Author: neztol
 * 
 * Created on 22 de abril de 2010, 10:31
 */

#include "CStatistics.h"

CStatistics::CStatistics() {
}

CStatistics::CStatistics(const CStatistics& orig) {
}

CStatistics::~CStatistics() {
}

void CStatistics::test(IplImage * imgBase, IplImage * imgRT, IplImage * imgBaseC, IplImage * imgRTC, t_Statistic_Item &item) {
    CViewMorphing vm(cvGetSize(imgBase));
    //CImageRegistration registration(cvGetSize(imgRT));

    IplImage * mask = cvCreateImage(cvGetSize(imgBase), IPL_DEPTH_8U, 1);

    cvSet(mask, cvScalar(255));

    //registration.registration(imgBase, NULL, imgRT);
    vm.viewMorphing(imgBase, imgRT, imgBaseC, imgRTC, mask, &item);

    vm.~CViewMorphing();
    cvReleaseImage(&mask);

}

void CStatistics::testChangingSize(string path1, string path2, CvSize currSize, t_Statistic_Item &item) {
    IplImage * imgBaseL = cvLoadImage(path1.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    IplImage * imgRTL = cvLoadImage(path2.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    IplImage * imgBaseCL = cvLoadImage(path1.c_str(), CV_LOAD_IMAGE_COLOR);
    IplImage * imgRTCL = cvLoadImage(path2.c_str(), CV_LOAD_IMAGE_COLOR);

    IplImage * imgBase = cvCreateImage(currSize, IPL_DEPTH_8U, 1);
    IplImage * imgRT = cvCreateImage(currSize, IPL_DEPTH_8U, 1);
    IplImage * imgBaseC = cvCreateImage(currSize, IPL_DEPTH_8U, 3);
    IplImage * imgRTC = cvCreateImage(currSize, IPL_DEPTH_8U, 3);

    cvResize(imgBaseL, imgBase);
    cvResize(imgRTL, imgRT);
    cvResize(imgBaseCL, imgBaseC);
    cvResize(imgRTCL, imgRTC);

    cvShowImage("Img1", imgBaseC);
    cvShowImage("Img2", imgRTC);

    test(imgBase, imgRT, imgBaseC, imgRTC, item);

    cvReleaseImage(&imgBase);
    cvReleaseImage(&imgRT);
    cvReleaseImage(&imgBaseC);
    cvReleaseImage(&imgRTC);

    cvReleaseImage(&imgBaseL);
    cvReleaseImage(&imgRTL);
    cvReleaseImage(&imgBaseCL);
    cvReleaseImage(&imgRTCL);
}

void CStatistics::testChangingParams(string path1, string path2, CvSize currSize, int zoom, int b1, int b2, t_Statistic_Item &item) {
    IplImage * imgBaseL = cvLoadImage(path1.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    IplImage * imgRTL = cvLoadImage(path2.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    IplImage * imgBaseCL = cvLoadImage(path1.c_str(), CV_LOAD_IMAGE_COLOR);
    IplImage * imgRTCL = cvLoadImage(path2.c_str(), CV_LOAD_IMAGE_COLOR);

    IplImage * imgBase = cvCreateImage(currSize, IPL_DEPTH_8U, 1);
    IplImage * imgRT = cvCreateImage(currSize, IPL_DEPTH_8U, 1);
    IplImage * imgBaseC = cvCreateImage(currSize, IPL_DEPTH_8U, 3);
    IplImage * imgRTC = cvCreateImage(currSize, IPL_DEPTH_8U, 3);    

#ifdef BAD_ASPECT
    CvRect rect = cvRect(246, 1, imgBaseL->width - 246, imgBaseL->height);
#else
    CvRect rect = cvRect(0, 0, imgBaseL->width, imgBaseL->height);
#endif
    
    if (zoom != 0) {
        int percentW = zoom * imgBaseL->width / 100;
        int percentH = zoom * imgBaseL->height / 100;
        rect = cvRect(rect.x + percentW, rect.y + percentH, rect.width - percentW, rect.height - percentH);
        cout << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << endl;        
        cvSetImageROI(imgBaseL, rect);
        cvSetImageROI(imgBaseCL, rect);
    }
    cvResize(imgBaseL, imgBase);
    cvResize(imgBaseCL, imgBaseC);
    cvResize(imgRTL, imgRT);
    cvResize(imgRTCL, imgRTC);

    cvReleaseImage(&imgBaseL);
    cvReleaseImage(&imgRTL);
    cvReleaseImage(&imgBaseCL);
    cvReleaseImage(&imgRTCL);

    //if (zoom != 0) {
    //    cvResetImageROI(imgBaseL);
    //    cvResetImageROI(imgBaseCL);
    //}

    if (b1 != 1) {
        cvSmooth(imgBase, imgBase, CV_GAUSSIAN, b1, b1);
        cvSmooth(imgBaseC, imgBaseC, CV_GAUSSIAN, b1, b1);
    }

    if (b2 != 1) {
        cvSmooth(imgRT, imgRT, CV_GAUSSIAN, b2, b2);
        cvSmooth(imgRTC, imgRTC, CV_GAUSSIAN, b2, b2);
    }

    cvShowImage("Img1", imgBaseC);
    cvShowImage("Img2", imgRTC);

    test(imgBase, imgRT, imgBaseC, imgRTC, item);

    cvReleaseImage(&imgBase);
    cvReleaseImage(&imgRT);
    cvReleaseImage(&imgBaseC);
    cvReleaseImage(&imgRTC);  //*/
}

bool CStatistics::isThisTested(string testName, CvSize currSize, int zoom, int b1, int b2) {
    bool result = true;
    sqlite3 * db;
    string dbName(PATH_BASE_STAT);
    dbName += "statistics.sqlite/statistics.sqlite";
    if (sqlite3_open(dbName.c_str(), &db) != SQLITE_OK){
        cerr << "Error al abrir la base de datos: " << sqlite3_errmsg(db) << endl;
    }

    sqlite3_stmt *statement;

    const char * sql = "SELECT count(nPtosMatch) FROM resultados where (width == ?) and (height == ?) and (zoom == ?) and (blur1 == ?) and (blur2 == ?) and (testName == ?)";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }
    sqlite3_bind_int(statement, 1, currSize.width);
    sqlite3_bind_int(statement, 2, currSize.height);
    sqlite3_bind_int(statement, 3, zoom);
    sqlite3_bind_int(statement, 4, b1);
    sqlite3_bind_int(statement, 5, b2);
    sqlite3_bind_text(statement, 6, testName.c_str(), testName.length(), NULL);
    if (sqlite3_step(statement) == SQLITE_ROW) {
        int nRows = sqlite3_column_int(statement, 0);
        cout << "nRows = " << nRows << endl;
        if (nRows == 0) {
            result = false;
        }
    }

    if (sqlite3_finalize(statement) != SQLITE_OK) {
        cerr << "Error al finalizar el statement" << endl;
    }

    if (sqlite3_close(db) != SQLITE_OK) {
        cerr << "Error al cerrar la base de datos: " << sqlite3_errmsg(db) << endl;
    }

    return result;
}

void CStatistics::saveResults(string testName, vector<t_Statistic_Item> items) {
    sqlite3 * db;
    string dbName = DB_PATH_BASE;
    if (sqlite3_open(dbName.c_str(), &db) != SQLITE_OK){
        cerr << "Error al abrir la base de datos: " << sqlite3_errmsg(db) << endl;
    }

    sqlite3_stmt *statement;
    const char * sql = "INSERT INTO resultados (testName, img1, img2, width, height, dist, ang, zoom, blur1, blur2, nPtosMatch, areaCubierta, pixelsDiferentes, resultImg, maskImg) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
    if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) != SQLITE_OK) {
        cerr << "Error al iniciar la consulta: " << sql << ", " << sqlite3_errmsg(db) << endl;
    }
    for (int i = 0; i < items.size(); i++) {
        t_Statistic_Item item = items.at(i);
        stringstream ss;
        ss << ".dist" << item.distance << ".angle" << item.angle << ".size" << item.size.width << "x" << item.size.height << ".zoom" << item.zoom << ".b1_" << item.blur1 << ".b2_" << item.blur2;

        //cout << ss.str() << endl;
        string pathMask = item.path + "." + ss.str() + ".mask.JPG";
        string pathRes = item.path + "." + ss.str() + ".result.JPG";

        //cout << pathMask << endl;
        //cout << pathRes << endl;//*/

        //cvSaveImage(pathMask.c_str(), item.mask);
        //cvSaveImage(pathRes.c_str(), item.result);

        //cvReleaseImage(&items.at(i).mask);
        //cvReleaseImage(&items.at(i).result);        

        sqlite3_bind_text(statement, 1, testName.c_str(), testName.length(), NULL);
        sqlite3_bind_text(statement, 2, item.pathBase.c_str(), item.pathBase.length(), NULL);
        sqlite3_bind_text(statement, 3, item.path.c_str(), item.path.length(), NULL);
        sqlite3_bind_int(statement, 4, item.size.width);
        sqlite3_bind_int(statement, 5, item.size.height);
        sqlite3_bind_double(statement, 6, item.distance);
        sqlite3_bind_double(statement,7, item.angle);
        sqlite3_bind_int(statement, 8, item.zoom);
        sqlite3_bind_int(statement, 9, item.blur1);
        sqlite3_bind_int(statement, 10, item.blur2);
        sqlite3_bind_int(statement, 11, item.nPuntosEmparejados);
        sqlite3_bind_int(statement, 12, item.areaCubierta);
        sqlite3_bind_int(statement, 13, item.pixelsDiferentes);
        sqlite3_bind_text(statement, 14, pathMask.c_str(), pathMask.length(), NULL);
        sqlite3_bind_text(statement, 15, pathRes.c_str(), pathRes.length(), NULL);

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

    if (sqlite3_close(db) != SQLITE_OK) {
        cerr << "Error al cerrar la base de datos: " << sqlite3_errmsg(db) << endl;
    }
}

void CStatistics::statistics() {
    int memIni = mem_total();

    string path(PATH_BASE_STAT);
    path += "datos.txt";
    string imgBasePath(PATH_BASE_STAT);

    ifstream ifs(path.c_str() , ifstream::in );

    char line[1024];
    ifs.getline(line, 1024);
    imgBasePath += line;
    imgBasePath += ".JPG";
    cout << "Imagen = " << imgBasePath << endl;
    ifs.getline(line, 1024);

    vector<t_Statistic_Item> items;

    path = string(PATH_BASE_STAT);
    while (ifs.good()) {
        t_Statistic_Item item;
        ifs >> item.distance;
        ifs >> item.angle;
        ifs.ignore(2);
        ifs.getline(line, 1024);
        item.path = path + line + ".JPG";
        item.pathBase = imgBasePath;
        items.push_back(item);
    }

    ifs.close();

    CvSize sizes[4];
    sizes[0] = cvSize(320, 240);
    sizes[1] = cvSize(640, 480);
    sizes[2] = cvSize(800, 600);

    cout << "Mem_final = " << mem_total() - memIni << "kB";
    exit(0);

    for (int z = 0; z <= 30; z +=5) {
        for (int s = 0; s < 3; s++) {
            for (int b1 = 1; b1 <= 9; b1 += 2) {
                for (int b2 = 1; b2 <= 9; b2 += 2) {

                    CvSize currSize = sizes[s];

                    string testName = TEST_NAME;

                    if (isThisTested(testName, currSize, z, b1, b2) == true) {
                        cout << "Ignorando " << currSize.width << "x" << currSize.height << ", z = " << z << ", b1 = " << b1 << ", b2 = " << b2 << endl;
                        continue;
                    } else {
                        cout << "Aceptando " << currSize.width << "x" << currSize.height << ", z = " << z << ", b1 = " << b1 << ", b2 = " << b2 << endl;
                    }

                    for (int i = 0; i < items.size(); i++) {
                        items.at(i).size = currSize;
                        items.at(i).zoom = z;
                        items.at(i).blur1 = b1;
                        items.at(i).blur2 = b2;

                        testChangingParams(imgBasePath, items.at(i).path, currSize, z, b1, b2, items.at(i));
                    }
                    
                    saveResults(testName, items);//*/
                }
            }
        }
    }
}

void CStatistics::tests(int testNumber) {
    string path(PATH_BASE_STAT);
    path += "datos.txt";
    string imgBasePath(PATH_BASE_STAT);

    ifstream ifs(path.c_str() , ifstream::in );

    char line[1024];
    ifs.getline(line, 1024);
    imgBasePath += line;
    imgBasePath += ".JPG";
    cout << "Imagen = " << imgBasePath << endl;
    ifs.getline(line, 1024);

    vector<t_Statistic_Item> items;

    path = string(PATH_BASE_STAT);
    while (ifs.good()) {
        t_Statistic_Item item;
        ifs >> item.distance;
        ifs >> item.angle;
        ifs.ignore(2);
        ifs.getline(line, 1024);
        item.path = path + line + ".JPG";
        item.pathBase = imgBasePath;
        items.push_back(item);
    }

    ifs.close();

    IplImage * imgDBL = cvLoadImage(imgBasePath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    IplImage * imgDBCL = cvLoadImage(imgBasePath.c_str(), CV_LOAD_IMAGE_COLOR);
    IplImage * imgDB = cvCreateImage(cvSize(800,600), IPL_DEPTH_8U, 1);
    IplImage * imgRT = cvCreateImage(cvSize(800,600), IPL_DEPTH_8U, 1);
    IplImage * imgDBC = cvCreateImage(cvSize(800,600), IPL_DEPTH_8U, 3);
    IplImage * imgRTC = cvCreateImage(cvSize(800,600), IPL_DEPTH_8U, 3);
    cvResize(imgDBL, imgDB);
    cvResize(imgDBCL, imgDBC);
    cvReleaseImage(&imgDBL);
    cvReleaseImage(&imgDBCL);
    CImageRegistration ir(cvGetSize(imgDB));
    for (int i = 0; i < items.size(); i++) {
        if (abs(items.at(i).distance) > 1) continue;
        if (abs(items.at(i).angle) > 10) continue;
        IplImage * imgRTL = cvLoadImage(items.at(i).path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        IplImage * imgRTCL = cvLoadImage(items.at(i).path.c_str(), CV_LOAD_IMAGE_COLOR);
        cvResize(imgRTL, imgRT);
        cvResize(imgRTCL, imgRTC);

        switch(testNumber) {
            case 0: {
                ir.getPairsOnBigImg(imgDB, imgRT, imgDBC, imgRTC);
                break;
            }
        }

        cvReleaseImage(&imgRTL);
        cvReleaseImage(&imgRTCL);
    }
    cvReleaseImage(&imgDB);
    cvReleaseImage(&imgDBC);
    cvReleaseImage(&imgRT);
    cvReleaseImage(&imgRTC);
}

void CStatistics::MRTP_test(CvSize size) {
    CMRPT_Route route1("/home/neztol/doctorado/Datos/MRPT_Data/malaga2009_parking_0L/", "GT_path_CAMERA_LEFT.txt");
    CMRPT_Route route2("/home/neztol/doctorado/Datos/MRPT_Data/malaga2009_parking_2L/", "GT_path_CAMERA_LEFT.txt");
    route2.addPoints("/home/neztol/doctorado/Datos/MRPT_Data/malaga2009_parking_6L/", "GT_path_CAMERA_LEFT.txt");

    t_RoutePoint currentPoint;

    IplImage * imgRT = cvCreateImage(size, IPL_DEPTH_8U, 1);
    IplImage * imgDB = cvCreateImage(size, IPL_DEPTH_8U, 1);

    CRealMatches rm(false, size);

    cvNamedWindow("imgRT", 1);
    cvNamedWindow("imgDB", 1);

    route1.setIndex(310);

    while (true) {
        IplImage * imgRT_L = route1.getNext(currentPoint);
        IplImage * imgDB_L = route2.getNearest(currentPoint);

        if (imgDB_L == NULL) {
            cvSet(imgDB, cvScalar(0, 0, 255));
        } else {
            cvResize(imgDB_L, imgDB);
        }

        cvResize(imgRT_L, imgRT);

        cvShowImage("imgRT", imgRT);
        cvShowImage("imgDB", imgDB);
        clock_t myTime = clock();
        rm.mainTest(imgRT, imgDB);
        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo TOTAL = " << time << endl;

        int key = cvWaitKey(0);
        if (key == 27)
            exit(0);
        if (key == 32)
            cvWaitKey(0);

        cvReleaseImage(&imgRT_L);
        cvReleaseImage(&imgDB_L);
    }

    cvReleaseImage(&imgRT);
    cvReleaseImage(&imgDB);
}

void CStatistics::statistics(int testIdx, int index, int z, int s, int b1, int b2) {    
    switch (testIdx) {
        case 0:
            PATH_BASE_STAT = "/home/neztol/doctorado/Datos/EstadisticasITER/ram/";
            DB_PATH_BASE = "/home/neztol/doctorado/Datos/EstadisticasITER/statistics.sqlite";
            PATH_BASE_IMG = "/home/neztol/doctorado/Datos/EstadisticasITER/ram/";
            TEST_NAME = "testITERtripode1";
            break;
        case 1:
            PATH_BASE_STAT = "/home/neztol/doctorado/Datos/EstadisticasITER/ram/";
            DB_PATH_BASE = "/home/neztol/doctorado/Datos/EstadisticasITER/statistics.sqlite";
            PATH_BASE_IMG = "/home/neztol/doctorado/Datos/EstadisticasITER/ram/";
            TEST_NAME = "testITERtripode2";
            break;
        case 2:
            PATH_BASE_STAT = "/home/neztol/doctorado/Datos/EstadisticasITER/ram/";
            DB_PATH_BASE = "/home/neztol/doctorado/Datos/EstadisticasITER/statistics.sqlite";
            PATH_BASE_IMG = "/home/neztol/doctorado/Datos/EstadisticasITER/ram/";
            TEST_NAME = "testITERtripode3";
            break;
    }


    string path(PATH_BASE_STAT);
    path += "datos.txt";
    string imgBasePath(PATH_BASE_IMG);

    ifstream ifs(path.c_str() , ifstream::in );

    char line[1024];
    ifs.getline(line, 1024);
    imgBasePath += line;
    imgBasePath += ".JPG";
    cout << "Imagen = " << imgBasePath << endl;
    ifs.getline(line, 1024);

    vector<t_Statistic_Item> items;

    path = string(PATH_BASE_STAT);
    while (ifs.good()) {
        t_Statistic_Item item;
        ifs >> item.distance;
        ifs >> item.angle;
        ifs.ignore(2);
        ifs.getline(line, 1024);
        item.path = path + line + ".JPG";
        item.pathBase = imgBasePath;
        items.push_back(item);
    }

    ifs.close();

    CvSize sizes[3];
    sizes[0] = cvSize(320, 240);
    sizes[1] = cvSize(640, 480);
    sizes[2] = cvSize(800, 600);


    CvSize currSize = sizes[s];

    string testName = TEST_NAME;

    if (isThisTested(testName, currSize, z, b1, b2) == true) {
        cout << "Ignorando " << currSize.width << "x" << currSize.height << ", z = " << z << ", b1 = " << b1 << ", b2 = " << b2 << endl;
        return;
    } else {
        cout << "Aceptando " << currSize.width << "x" << currSize.height << ", z = " << z << ", b1 = " << b1 << ", b2 = " << b2 << endl;
    }//*/

    t_Statistic_Item item = items.at(index);

    item.size = currSize;
    item.zoom = z;
    item.blur1 = b1;
    item.blur2 = b2;        

    testChangingParams(imgBasePath, item.path, currSize, z, b1, b2, item);

    items.clear();
    items.push_back(item);

    saveResults(testName, items);//*/

    //cvWaitKey(0);
}
