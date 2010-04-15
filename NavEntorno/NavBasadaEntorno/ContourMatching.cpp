#include "ViewMorphing.h"

typedef struct {
    CvSeq * contour1;
    CvSeq * contour2;

    CvBox2D box1;
    CvBox2D box2;

    double distance;

    double moment;
} t_ContourMatch;

#define CANNY_SOLO 0
#define CANNY_DILATADO 0
#define MODO_PREPROCESADO CANNY_SOLO

void CViewMorphing::contourMatching(IplImage * img1, IplImage * &img2) {

    /*IplImage * tmp1 = cvCreateImage(cvSize(img1->width * 2, img1->height * 2), IPL_DEPTH_8U, 1);
    IplImage * tmp2 = cvCreateImage(cvSize(img2->width * 2, img2->height * 2), IPL_DEPTH_8U, 1);
    cvResize(img1, tmp1, CV_INTER_CUBIC);
    cvResize(img2, tmp2, CV_INTER_CUBIC);
    cvReleaseImage(&img1);
    cvReleaseImage(&img2);
    img1 = tmp1;
    img2 = tmp2;*/

    cvNamedWindow("Canny1");
    cvNamedWindow("Canny2");
    cvNamedWindow("Contour1");
    cvNamedWindow("Contour2");
    cvNamedWindow("ContourA");
    cvNamedWindow("ContourB");

    IplImage * canny1 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
    IplImage * canny2 = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 1);    
    IplImage * contours1 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);
    IplImage * contours2 = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 3);

    if (MODO_PREPROCESADO == CANNY_SOLO) {
        cvCanny(img1, canny1, 200, 255, 3);
        cvCanny(img2, canny2, 200, 255, 3);
    } else if (MODO_PREPROCESADO == CANNY_DILATADO) {
        cvSmooth(img1, canny1, CV_GAUSSIAN, 7, 7);
        cvSmooth(img2, canny2, CV_GAUSSIAN, 7, 7);
        cvCanny(canny1, canny1, 200, 255, 3);
        cvCanny(canny2, canny2, 200, 255, 3);
        cvDilate(canny1, canny1, 0, 3);
        cvDilate(canny2, canny2, 0, 3);
    }
    
    cvShowImage("Canny1", canny1);
    cvShowImage("Canny2", canny2);

    //if (cvWaitKey(0) == 27) exit(0);

    CvMemStorage* storage1 = cvCreateMemStorage(0);
    CvSeq* contour1 = 0;
    CvMemStorage* storage2 = cvCreateMemStorage(0);
    CvSeq* contour2 = 0;

    cvFindContours(canny1, storage1, &contour1, sizeof (CvContour), CV_RETR_LIST, CV_LINK_RUNS);
    cvFindContours(canny2, storage2, &contour2, sizeof (CvContour), CV_RETR_LIST, CV_LINK_RUNS);

    cvZero(contours1);
    cvZero(contours2);

    for (; contour1 != 0; contour1 = contour1->h_next) {
        CvScalar color = CV_RGB(rand()&255, rand()&255, rand()&255);
        cvDrawContours(contours1, contour1, color, color, -1, 1, 8);
    }
    for (; contour2 != 0; contour2 = contour2->h_next) {
        CvScalar color = CV_RGB(rand()&255, rand()&255, rand()&255);
        cvDrawContours(contours2, contour2, color, color, -1, 1, 8);
    }

    cvShowImage("ContourA", contours1);
    cvShowImage("ContourB", contours2);

    cvFindContours(canny1, storage1, &contour1, sizeof (CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    cvFindContours(canny2, storage2, &contour2, sizeof (CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    cvCvtColor(img1, contours1, CV_GRAY2BGR);
    cvCvtColor(img2, contours2, CV_GRAY2BGR);

    /*for( ; contour1 != 0; contour1 = contour1->h_next ) {
        //if ( contour1->total < 40) continue;
        if (cvContourPerimeter(contour1) < 100) continue;
        CvSeq * minErrSeq = NULL;
        double minErr = 9999999;
        double area1 = abs(cvContourArea(contour1));
        for(CvSeq * tmpContour = contour2; tmpContour != 0; tmpContour = tmpContour->h_next ) {
            //if ( contour2->total < 40) continue;
            if (cvContourPerimeter(tmpContour) < 100) continue;
            double area2 = abs(cvContourArea(tmpContour));
            if (min(area1, area2) / max(area1, area2) < 0.80) continue;

            //double match_error = cvMatchShapes(contour1, tmpContour, CV_CONTOURS_MATCH_I3, 0);
            double match_error = pghMatchShapes(contour1, tmpContour);
            //double match_error = treeMatchShapes(contour1, tmpContour);

            if (match_error < minErr) {
                minErr = match_error;
                minErrSeq = tmpContour;
            }
        }

        if (minErrSeq != NULL) {
        //if (minErr < (0.75-threshold*.75)) {
            CvScalar color = CV_RGB( rand()&255, rand()&255, rand()&255 );
            double area2 = abs(cvContourArea(minErrSeq));
            double prop = min(area1, area2) / max(area1, area2);
            CvBox2D  box1 = cvMinAreaRect2(contour1);
            CvBox2D  box2 = cvMinAreaRect2(minErrSeq);
            cout << "match_error = " << minErr << ", prop = " << prop << ", ang1 = " << box1.angle << ", ang2 = " << box2.angle << endl;

            if (abs(box1.angle - box2.angle) < 10) {
                drawBox2D(contours1, box1);
                drawBox2D(contours2, box2);
                cvDrawContours(contours1, contour1, color, color, -1, 1, 8 );
                cvDrawContours(contours2, minErrSeq, color, color, -1, 1, 8 );
            }

            cvShowImage("Contour1", contours1);
            cvShowImage("Contour2", contours2);

            if (cvWaitKey(0) == 27) exit(0);
            
         //}
        }
    }*/

    /*for (; contour1 != 0; contour1 = contour1->h_next) {
        if (cvContourPerimeter(contour1) < 100) continue;

        CvSeq * minErrSeq = NULL;
        double minErr = DBL_MAX;
        CvBox2D box1 = cvMinAreaRect2(contour1);

        for (CvSeq * tmpContour = contour2; tmpContour != 0; tmpContour = tmpContour->h_next) {
            if (cvContourPerimeter(tmpContour) < 100) continue;
            
            CvBox2D box2 = cvMinAreaRect2(tmpContour);
            //double area2 = box2.size.width * box2.size.height;

            //if (min(area1, area2) / max(area1, area2) < 0.80) continue;
            cout << "Width = " << min(box1.size.width, box2.size.width) / max(box1.size.width, box2.size.width);
            cout << ", Height = " << min(box1.size.height, box2.size.height) / max(box1.size.height, box2.size.height) << endl;
            cout << "difAng = " << abs(box1.angle - box2.angle) << endl;
            
            if (min(box1.size.width, box2.size.width) / max(box1.size.width, box2.size.width) > 0.50) continue;
            if (min(box1.size.height, box2.size.height) / max(box1.size.height, box2.size.height) > 0.50) continue;
            if (abs(box1.angle - box2.angle) > 10) continue;

            cout << "Width = " << min(box1.size.width, box2.size.width) / max(box1.size.width, box2.size.width);
            cout << ", Height = " << min(box1.size.height, box2.size.height) / max(box1.size.height, box2.size.height) << endl;
            cout << "difAng = " << abs(box1.angle - box2.angle) << endl;

            //double match_error = cvMatchShapes(contour1, tmpContour, CV_CONTOURS_MATCH_I3, 0);
            double match_error = pghMatchShapes(contour1, tmpContour);
            //double match_error = treeMatchShapes(contour1, tmpContour);
            cout << "match_error = " << match_error << endl;

            if (match_error < minErr) {
                minErr = match_error;
                minErrSeq = tmpContour;
            }
        }

        if (minErr <= 100) {
            cout << "minErr " << minErr << endl;
            CvScalar color = CV_RGB(rand()&255, rand()&255, rand()&255);
            drawBox2D(contours1, box1);
            CvBox2D box2 = cvMinAreaRect2(minErrSeq);
            drawBox2D(contours2, box2);
            cvDrawContours(contours1, contour1, color, color, -1, 1, 8);
            cvDrawContours(contours2, minErrSeq, color, color, -1, 1, 8);
            
            cvShowImage("Contour1", contours1);
            cvShowImage("Contour2", contours2);

            if (cvWaitKey(0) == 27) exit(0);
        }
    }*/

    cout << "Entro " << endl;
    int nContours1 = 0, nContours2 = 0;
    for (CvSeq * counter = contour1; counter != 0; counter = counter->h_next, nContours1++);
    for (CvSeq * counter = contour2; counter != 0; counter = counter->h_next, nContours2++);

    t_ContourMatch ** matching = new t_ContourMatch*[nContours1];
    for (int i = 0; i < nContours1; i++)
        matching[i] = new t_ContourMatch[nContours2];
    for (int i = 0; contour1 != 0; contour1 = contour1->h_next, i++) {
        if (cvContourPerimeter(contour1) < 100) {
            for (int j = 0; j < nContours2; j++) {
                matching[i][j].moment = -1;
            }
            continue;
        }

        CvBox2D box1 = cvMinAreaRect2(contour1);

        int j = 0;
        for (CvSeq * tmpContour = contour2; tmpContour != 0; tmpContour = tmpContour->h_next, j++) {
            if (cvContourPerimeter(tmpContour) < 100) {
                matching[i][j].moment = -1;
                continue;
            }

            CvBox2D box2 = cvMinAreaRect2(tmpContour);

            matching[i][j].contour1 = contour1;
            matching[i][j].contour2 = tmpContour;
            matching[i][j].box1 = box1;
            matching[i][j].box2 = box2;

            matching[i][j].distance = sqrt(pow(box1.center.x - box2.center.x, 2.0) + pow(box1.center.y - box2.center.y, 2.0));

            //matching[i][j].moment = cvMatchShapes(contour1, tmpContour, CV_CONTOURS_MATCH_I3, 0);
            matching[i][j].moment = pghMatchShapes(contour1, tmpContour);
            //matching[i][j].moment = treeMatchShapes(contour1, tmpContour);         
        }
    }

    while (true) {
        //cvZero(contours1);
        //cvZero(contours2);

        t_ContourMatch maxMatch;
        maxMatch.moment = DBL_MAX;
        int posI = -1, posJ = -1;
        for (int i = 0; i < nContours1; i++) {
            for (int j = 0; j < nContours2; j++) {
                if ((matching[i][j].moment != -1) && (matching[i][j].moment < maxMatch.moment)) {
                    maxMatch.moment = matching[i][j].moment;
                    maxMatch.contour1 = matching[i][j].contour1;
                    maxMatch.contour2 = matching[i][j].contour2;
                    maxMatch.box1 = matching[i][j].box1;
                    maxMatch.box2 = matching[i][j].box2;
                    maxMatch.distance = matching[i][j].distance;
                    posI = i;
                    posJ = j;
                }
            }
        }
        if (maxMatch.moment == DBL_MAX) break;
        matching[posI][posJ].moment = -1;

        if (MODO_PREPROCESADO == CANNY_SOLO) {
            //if (maxMatch.moment > 7) break;
            if (abs(maxMatch.box1.angle - maxMatch.box2.angle) > 7) {
                cout << "Angulo invalido, momento = " << maxMatch.moment << endl;
                continue;
            }
            if (maxMatch.distance > 30) {
                cout << "Dist invalido, momento = " << maxMatch.moment << endl;
                continue;
            }
        } else if (MODO_PREPROCESADO == CANNY_DILATADO) {
            //if (maxMatch.moment > 7) break;
            if (abs(maxMatch.box1.angle - maxMatch.box2.angle) > 7) {
                cout << "Angulo invalido, momento = " << maxMatch.moment << endl;
                continue;
            }
            /*if (maxMatch.distance > 100) {
                cout << "Dist invalido, momento = " << maxMatch.moment << endl;
                continue;
            }*/
        }

        // Como lo damos por válido, eliminamos la competencia
        for (int i = 0; i < nContours1; i++) matching[i][posJ].moment = -1;
        for (int j = 0; j < nContours1; j++) matching[posI][j].moment = -1;

        CvScalar color = CV_RGB(rand()&255, rand()&255, rand()&255);
        drawBox2D(contours1, maxMatch.box1);
        drawBox2D(contours2, maxMatch.box2);
        cvDrawContours(contours1, maxMatch.contour1, color, color, -1, 1, 8);
        cvDrawContours(contours2, maxMatch.contour2, color, color, -1, 1, 8);

        cout << "Width = " << min(maxMatch.box1.size.width, maxMatch.box2.size.width) / max(maxMatch.box1.size.width, maxMatch.box2.size.width);
        cout << ", Height = " << min(maxMatch.box1.size.height, maxMatch.box2.size.height) / max(maxMatch.box1.size.height, maxMatch.box2.size.height) << endl;
        cout << "difAng = " << abs(maxMatch.box1.angle - maxMatch.box2.angle) << endl;
        cout << "distance = " << maxMatch.distance << endl;

        double myMoment = cvMatchShapes(maxMatch.contour1, maxMatch.contour2, CV_CONTOURS_MATCH_I3, 0);
        double myMoment2 = pghMatchShapes(maxMatch.contour1, maxMatch.contour2);
        //double myMoment3 = treeMatchShapes(maxMatch.contour1, maxMatch.contour2);

        cout << "mmtShapes = " << myMoment << ", mmtHist = " << myMoment2 << ", mmtTree = " << -1 << endl;

        /*cvShowImage("Contour1", contours1);
        cvShowImage("Contour2", contours2);

        if (cvWaitKey(0) == 27) break;*/
    }
    cout << "Salio" << endl;

    cvShowImage("Contour1", contours1);
    cvShowImage("Contour2", contours2);

    cvReleaseImage(&canny1);
    cvReleaseImage(&canny2);
    cvReleaseImage(&contours1);
    cvReleaseImage(&contours2);

    if (cvWaitKey(1000) == 27) exit(0);
}

// http://code.google.com/p/eyepatch/source/browse/trunk/ShapeClassifier.cpp

double CViewMorphing::pghMatchShapes(CvSeq *shape1, CvSeq *shape2) {
    int dims[] = {8, 8};
    float range[] = {-180, 180, -100, 100};
    float *ranges[] = {&range[0], &range[2]};
    CvHistogram* hist1 = cvCreateHist(2, dims, CV_HIST_ARRAY, ranges, 1);
    CvHistogram* hist2 = cvCreateHist(2, dims, CV_HIST_ARRAY, ranges, 1);
    cvCalcPGH(shape1, hist1);
    cvCalcPGH(shape2, hist2);
    cvNormalizeHist(hist1, 100.0f);
    cvNormalizeHist(hist2, 100.0f);
    double corr = cvCompareHist(hist1, hist2, CV_COMP_CHISQR);
    cvReleaseHist(&hist1);
    cvReleaseHist(&hist2);
    return corr;
}

double CViewMorphing::treeMatchShapes(CvSeq *shape1, CvSeq *shape2) {
    CvMemStorage* storage1 = cvCreateMemStorage(0);
    CvMemStorage* storage2 = cvCreateMemStorage(0);

    CvContourTree * tree1 = cvCreateContourTree(shape1, storage1, 0);
    CvContourTree * tree2 = cvCreateContourTree(shape2, storage2, 0);

    return cvMatchContourTrees(tree1, tree2, CV_CONTOUR_TREES_MATCH_I1, 0);
}

void CViewMorphing::drawBox2D(IplImage * img, CvBox2D box) {
    CvPoint2D32f pt[4];

    cvBoxPoints(box, pt);

    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;
        cvLine(img, cvPointFrom32f(pt[i]), cvPointFrom32f(pt[j]), cvScalar(0, 255, 0));
    }

}
