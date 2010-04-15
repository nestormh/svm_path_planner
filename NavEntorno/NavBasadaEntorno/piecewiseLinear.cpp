#include "CRealMatches.h"

inline void CRealMatches::drawDelaunay(char * name1, char * name2, CvSubdiv2D * subdiv, IplImage * img, CvSize size, CvPoint2D32f currPoint) {

    CvSeqReader reader;
    int i, total = subdiv->edges->total;
    int elem_size = subdiv->edges->elem_size;

    cvStartReadSeq((CvSeq*) (subdiv->edges), &reader, 0);

    IplImage * imgDelaunay = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * imgDelaunay2 = cvCreateImage(size, IPL_DEPTH_8U, 3);
    cvCvtColor(img, imgDelaunay, CV_GRAY2BGR);
    cvZero(imgDelaunay2);
    cvCvtColor(img, imgDelaunay2, CV_GRAY2BGR);
    cvNamedWindow(name1, 1);
    cvNamedWindow(name2, 1);

    cvSetImageCOI(imgDelaunay2, 1);
    cvCopyImage(mask2, imgDelaunay2);
    cvSetImageCOI(imgDelaunay2, 0);

    for (i = 0; i < total; i++) {
        CvQuadEdge2D* edge = (CvQuadEdge2D*) (reader.ptr);

        if (CV_IS_SET_ELEM(edge)) {

            CvPoint2D32f p1OrgF = cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt;
            CvPoint2D32f p1DestF = cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge)->pt;

            CvPoint org = cvPointFrom32f(p1OrgF);
            CvPoint dest = cvPointFrom32f(p1DestF);

            //cvLine(imgDelaunay2, org, dest, cvScalar(255));

            CvSubdiv2DEdge edge1 = cvSubdiv2DGetEdge((CvSubdiv2DEdge) edge, CV_PREV_AROUND_RIGHT);
            CvSubdiv2DEdge edge2 = cvSubdiv2DGetEdge((CvSubdiv2DEdge) edge1, CV_NEXT_AROUND_LEFT);

            cvLine(imgDelaunay, dest, cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt), cvScalar(255, 0, 255));
            cvLine(imgDelaunay, cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt), cvPointFrom32f(cvSubdiv2DEdgeDst(edge2)->pt), cvScalar(0, 255, 255));
            cvCircle(imgDelaunay, org, 2, cvScalar(255, 0, 0), -1);
            cvCircle(imgDelaunay, dest, 2, cvScalar(255, 0, 0), 1);

            CvPoint * puntos = new CvPoint[3];
            int nPuntos = 3;
            puntos[0] = org;
            puntos[1] = dest;
            puntos[2] = cvPointFrom32f(cvSubdiv2DEdgeDst(edge1)->pt);

            cvFillConvexPoly(imgDelaunay, &puntos[0], nPuntos, cvScalar(0, 0, 255)); //CV_RGB(rand()&255,rand()&255,rand()&255));
            cvPolyLine(imgDelaunay2, &(puntos), &nPuntos, 1, 1, cvScalarAll(255));
        }        

        CV_NEXT_SEQ_ELEM(elem_size, reader);
    }

    cvCircle(imgDelaunay2, cvPointFrom32f(currPoint), 2, cvScalar(0, 0, 255), -1);

    cvShowImage(name1, imgDelaunay);
    cvShowImage(name2, imgDelaunay2);

    cvReleaseImage(&imgDelaunay);
    cvReleaseImage(&imgDelaunay2);
}

inline void CRealMatches::getTriangle(CvSubdiv2D * subdiv, CvPoint2D32f point, CvPoint2D32f * &tri) {
    /*CvSubdiv2DEdge edge = 0;
    CvSubdiv2DPoint* p = 0;
    cvSubdiv2DLocate(subdiv, point, &edge, &p);

    CvPoint2D32f org = cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge)->pt;
    CvPoint2D32f dest = cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge)->pt;

    CvSubdiv2DEdge edge1 = cvSubdiv2DGetEdge((CvSubdiv2DEdge) edge, CV_PREV_AROUND_LEFT);

    tri[0] = org;
    tri[1] = dest;
    tri[2] = cvSubdiv2DEdgeOrg(edge1)->pt;*/


    //TODO: Copiar de la implementación de la obtención de los triangulos en el PL
    CvSubdiv2DEdge e;
    CvSubdiv2DEdge e0;
    CvSubdiv2DPoint * p = 0;

    cvSubdiv2DLocate(subdiv, point, &e0, &p);

    int index = 0;

    if (e0) {
        e = e0;
        bool toTransform = true;
        do {
            CvPoint2D32f vertex = cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) e)->pt;

            if (vertex.x < 0 || vertex.y < 0 || vertex.x >= size.width || vertex.y >= size.height) {
                tri[0] = cvPoint2D32f(-1, -1);
                tri[1] = cvPoint2D32f(-1, -1);
                tri[2] = cvPoint2D32f(-1, -1);
                return;
            }

            tri[index] = vertex;
            index++;

            e = cvSubdiv2DGetEdge(e, CV_NEXT_AROUND_LEFT);
        } while (e != e0);
    } else {
        tri[0] = cvPoint2D32f(-1, -1);
        tri[1] = cvPoint2D32f(-1, -1);
        tri[2] = cvPoint2D32f(-1, -1);
        return;
    }
}

void CRealMatches::drawTriangles(char * name1, char * name2, bool originPoints) {
    CvRect rect = cvRect(0, 0, size.width * 2, size.height * 2);
    CvMemStorage * storage = cvCreateMemStorage(0);

    CvSubdiv2D * subdiv = cvCreateSubdivDelaunay2D(rect, storage);
    for (vector<t_Pair>::iterator it = pairs.begin(); it != pairs.end(); it++) {
        if ((it->p1.x < 0) || (it->p1.y < 0) ||
                (it->p1.x > size.width - 1) || (it->p1.y > size.height - 1)) {
            continue;
        }
        if ((it->p2.x < 0) || (it->p2.y < 0) ||
                (it->p2.x > size.width - 1) || (it->p2.y > size.height - 1)) {
            continue;
        }

        try {
            if (originPoints) {
                cvSubdivDelaunay2DInsert(subdiv, it->p1);
            } else {
                cvSubdivDelaunay2DInsert(subdiv, it->p2);
            }
        } catch (cv::Exception e) {
        }
    }

    if (originPoints) {
        drawDelaunay(name1, name2, subdiv, img1, size, cvPoint2D32f(-10, -10));
    } else {
        drawDelaunay(name1, name2, subdiv, img2, size, cvPoint2D32f(-10, -10));
    }
    cvReleaseMemStorage(&storage);
}

void CRealMatches::pieceWiseLinear() {
    CvSize size = cvGetSize(img1);
    // Primero automatizamos las correspondencias entre los puntos
    CvMat * mCorrespPoints = cvCreateMat(size.height, size.width, CV_64FC2);
    cvZero(mCorrespPoints);

    CvMat * A = cvCreateMat(3, 3, CV_64FC1);
    CvMat * B = cvCreateMat(3, 3, CV_64FC1);
    CvMat * C = cvCreateMat(3, 3, CV_64FC1);
    CvMat * D = cvCreateMat(3, 3, CV_64FC1);

    cvSet(A, cvScalar(1));
    cvSet(B, cvScalar(1));
    cvSet(C, cvScalar(1));
    cvSet(D, cvScalar(1));

    CvRect rect = cvRect(0, 0, size.width * 2, size.height * 2);
    CvMemStorage * storage = cvCreateMemStorage(0);

    clock_t myTime = clock();
    CvSubdiv2D * subdiv = cvCreateSubdivDelaunay2D(rect, storage);
    for (vector<t_Pair>::iterator it = pairs.begin(); it != pairs.end(); it++) {
        if ((it->p1.x < 0) || (it->p1.y < 0) ||
                (it->p1.x > size.width - 1) || (it->p1.y > size.height - 1)) {            
            continue;
        }
        if ((it->p2.x < 0) || (it->p2.y < 0) ||
                (it->p2.x > size.width - 1) || (it->p2.y > size.height - 1)) {
            continue;
        }

        try {
            cvSubdivDelaunay2DInsert(subdiv, it->p1);
            cvSet2D(mCorrespPoints, it->p1.y, it->p1.x, cvScalar(it->p2.x, it->p2.y));
        } catch (cv::Exception e) {
        }
    }
    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en Delaunay = " << time << endl;

    CvMat * remapX = cvCreateMat(size.height, size.width, CV_32FC1);
    CvMat * remapY = cvCreateMat(size.height, size.width, CV_32FC1);

    CvPoint2D32f myPoint;

    CvSubdiv2DEdge e;
    CvSubdiv2DEdge e0;
    CvSubdiv2DPoint * p = 0;

    CvPoint2D32f point1[3];
    CvPoint2D32f point2[3];
    
    for (int x = 0; x < size.width; x++) {
        for (int y = 0; y < size.height; y++) {            
            if (cvGetReal2D(mask1, y, x) != 255) continue;

            myPoint = cvPoint2D32f(x, y);
            
            cvSubdiv2DLocate(subdiv, myPoint, &e0, &p);

            int index = 0;

            if (e0) {
                e = e0;
                bool toTransform = true;
                do {
                    CvPoint2D32f p1 = cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) e)->pt;
                    CvPoint2D32f p2;

                    if (p1.x < 0 || p1.y < 0 || p1.x >= size.width || p1.y >= size.height) {
                        toTransform = false;
                        break; //p2 = p1;
                    } else {
                        CvScalar p2Val = cvGet2D(mCorrespPoints, cvRound(p1.y), cvRound(p1.x));

                        p2 = cvPoint2D32f(p2Val.val[0], p2Val.val[1]);
                    }

                    point1[index] = p1;
                    point2[index] = p2;
                    index++;

                    e = cvSubdiv2DGetEdge(e, CV_NEXT_AROUND_LEFT);
                } while (e != e0);
                if (toTransform == false)
                    continue;
            } else {
                //cout << "No se encontro e0" << endl;
                continue;
            }

            // Ahora vamos a calcular el punto central que nos deber�a quedar
            // Calculamos A
            for (int j = 0; j < 3; j++) {
                cvmSet(A, j, 0, point1[j].y);
                cvmSet(A, j, 1, point2[j].x);
                cvmSet(B, j, 0, point1[j].x);
                cvmSet(B, j, 1, point2[j].x);
                cvmSet(C, j, 0, point1[j].x);
                cvmSet(C, j, 1, point1[j].y);
                cvmSet(D, j, 0, point1[j].x);
                cvmSet(D, j, 1, point1[j].y);
                cvmSet(D, j, 2, point2[j].x);
            }

            double a = cvDet(A);
            double b = -cvDet(B);
            double c = cvDet(C);
            double d = -cvDet(D);

            CvPoint2D32f newPos = cvPoint2D32f(0, 0);
            newPos.x = (-a * myPoint.x - b * myPoint.y - d) / c;

            // Calculamos la componente y
            for (int j = 0; j < 3; j++) {
                cvmSet(A, j, 0, point1[j].y);
                cvmSet(A, j, 1, point2[j].y);
                cvmSet(B, j, 0, point1[j].x);
                cvmSet(B, j, 1, point2[j].y);
                cvmSet(C, j, 0, point1[j].x);
                cvmSet(C, j, 1, point1[j].y);
                cvmSet(D, j, 0, point1[j].x);
                cvmSet(D, j, 1, point1[j].y);
                cvmSet(D, j, 2, point2[j].y);
            }

            a = cvDet(A);
            b = -cvDet(B);
            c = cvDet(C);
            d = -cvDet(D);

            newPos.y = (-a * myPoint.x - b * myPoint.y - d) / c;

            cvSetReal2D(remapX, cvRound(myPoint.y), cvRound(myPoint.x), newPos.x);
            cvSetReal2D(remapY, cvRound(myPoint.y), cvRound(myPoint.x), newPos.y);
        }
    }

    for (vector<t_Pair>::iterator it = pairs.begin(); it != pairs.end(); it++) {
        //CvScalar val = cvGet2D(img2, )
        if ((it->p1.x < 0) || (it->p1.y < 0) ||
                (it->p1.x > size.width - 1) || (it->p1.y > size.height - 1)) {
            continue;
        }
        if ((it->p2.x < 0) || (it->p2.y < 0) ||
                (it->p2.x > size.width - 1) || (it->p2.y > size.height - 1)) {
            continue;
        }
        cvSetReal2D(remapX, it->p1.y, it->p1.x, it->p2.x);
        cvSetReal2D(remapY, it->p1.y, it->p1.x, it->p2.y);
    }//*/

    cvRemap(img2, plinear, remapX, remapY, CV_INTER_CUBIC);
    cvSet(mask2, cvScalar(255));
    cvRemap(mask2, mask2, remapX, remapY, CV_WARP_FILL_OUTLIERS + CV_INTER_CUBIC);

    cvReleaseMat(&A);
    cvReleaseMat(&B);
    cvReleaseMat(&C);
    cvReleaseMat(&D);
    cvReleaseMat(&mCorrespPoints);
    cvReleaseMat(&remapX);
    cvReleaseMat(&remapY);
    cvReleaseMemStorage(&storage);
}

void CRealMatches::cleanByTriangles() {
    CvRect rect = cvRect(0, 0, size.width * 2, size.height * 2);
    CvMemStorage * storage1 = cvCreateMemStorage(0);
    CvMemStorage * storage2 = cvCreateMemStorage(0);
    CvMat * mCorrespPoints = cvCreateMat(size.height, size.width, CV_64FC2);
    cvZero(mCorrespPoints);
    vector<bool> similarTriangles;

    CvSubdiv2D * subdiv1 = cvCreateSubdivDelaunay2D(rect, storage1);
    CvSubdiv2D * subdiv2 = cvCreateSubdivDelaunay2D(rect, storage2);
    for (vector<t_Pair>::iterator it = pairs.begin(); it != pairs.end(); it++) {
        if ((it->p1.x < 0) || (it->p1.y < 0) ||
                (it->p1.x > size.width - 1) || (it->p1.y > size.height - 1)) {
            continue;
        }
        if ((it->p2.x < 0) || (it->p2.y < 0) ||
                (it->p2.x > size.width - 1) || (it->p2.y > size.height - 1)) {
            continue;
        }

        try {
            cvSubdivDelaunay2DInsert(subdiv1, it->p1);
            cvSubdivDelaunay2DInsert(subdiv2, it->p2);
            cvSet2D(mCorrespPoints, it->p1.y, it->p1.x, cvScalar(it->p2.x, it->p2.y));            
        } catch (cv::Exception e) {
        }
    }

    CvPoint2D32f * tri1 = new CvPoint2D32f[3];
    CvPoint2D32f * tri2 = new CvPoint2D32f[3];
    for (vector<t_Pair>::iterator it = pairs.begin(); it != pairs.end(); it++) {
        getTriangle(subdiv1, it->p1, tri1);
        getTriangle(subdiv2, it->p2, tri2);
        if (!(((tri1[0].x == tri2[0].x) && (tri1[0].y == tri2[0].y) &&
             (tri1[1].x == tri2[1].x) && (tri1[1].y == tri2[1].y) &&
             (tri1[2].x == tri2[2].x) && (tri1[2].y == tri2[2].y)) ||

            ((tri1[0].x == tri2[2].x) && (tri1[0].y == tri2[2].y) &&
             (tri1[1].x == tri2[0].x) && (tri1[1].y == tri2[0].y) &&
             (tri1[2].x == tri2[1].x) && (tri1[2].y == tri2[1].y)) ||

            ((tri1[0].x == tri2[1].x) && (tri1[0].y == tri2[1].y) &&
             (tri1[1].x == tri2[2].x) && (tri1[1].y == tri2[2].y) &&
             (tri1[2].x == tri2[0].x) && (tri1[2].y == tri2[0].y)))) {

            pairs.erase(it);
            if (it == pairs.end())
                break;
        }        
    }

    cvReleaseMemStorage(&storage1);
    cvReleaseMemStorage(&storage2);
    cvReleaseMat(&mCorrespPoints);
}