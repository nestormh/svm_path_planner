// http://n2.nabble.com/cvExtractSURF-td2126985.html

#include "ImageRegistration.h"

#define EXTENDED_DESCRIPTOR 1
#define CORRELATION_THRESHOLD 0.7
#define CORRELATION_THRESHOLD_MESR 0.6

//void bruteMatch(IplImage * img1, IplImage * img2, CvMat **points1, CvMat **points2, CvSeq *kp1, CvSeq *desc1, CvSeq *kp2, CvSeq * desc2);
//void showPairs(char * name, IplImage * img1, IplImage * img2, CvMat * points1, CvMat * points2);
//void removeOutliers(CvMat **points1, CvMat **points2, CvMat *status);

void pruebaSurf(IplImage * img1, IplImage * img2) {
    clock_t myTime = clock();

    CvSeq *kp1 = NULL, *kp2 = NULL;
    CvSeq *desc1 = NULL, *desc2 = NULL;
    CvMemStorage *storage = cvCreateMemStorage(0);
    cvExtractSURF(img1, NULL, &kp1, &desc1, storage, cvSURFParams(600, EXTENDED_DESCRIPTOR));
    cvExtractSURF(img2, NULL, &kp2, &desc2, storage, cvSURFParams(600, EXTENDED_DESCRIPTOR));

    CvMat * points1, * points2;

    bruteMatch(img1, img2, &points1, &points2, kp1, desc1, kp2, desc2);

    CvMat * F = cvCreateMat(3, 3, CV_32FC1);
    CvMat *status = cvCreateMat(1, points1->cols, CV_8UC1);
    int fm_count = cvFindFundamentalMat(points1, points2, F, CV_FM_RANSAC, 1., 0.99, status);
    removeOutliers(&points1, &points2, status);

    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en surf = " << time << endl;


    showPairs2("surf", img1, img2, points1, points2);
}

/* You will have to correlate the descriptors with each other to determine
which keypoints in each rectangle corresponds to one another. You could use
a BBF tree which is implemented in the latest version of OpenCV, but unless
your rectangle is huge, you might just as well just correlate them the
standard way, which I do like this :*/

// brute-force attempt at correlating the two sets of features
void bruteMatch(IplImage * img1, IplImage * img2, CvMat **points1, CvMat **points2, CvSeq *kp1, CvSeq *desc1, CvSeq *kp2, CvSeq * desc2) {
    int i, j, k;

    double* avg1 = (double*) malloc(sizeof (double) * kp1->total);
    double* avg2 = (double*) malloc(sizeof (double) * kp2->total);
    double* dev1 = (double*) malloc(sizeof (double) * kp1->total);
    double* dev2 = (double*) malloc(sizeof (double) * kp2->total);

    int* best1 = (int*) malloc(sizeof (int) * kp1->total);
    int* best2 = (int*) malloc(sizeof (int) * kp2->total);

    double* best1corr = (double*) malloc(sizeof (double) * kp1->total);
    double* best2corr = (double*) malloc(sizeof (double) * kp2->total);

    float *seq1, *seq2;
    int descriptor_size = EXTENDED_DESCRIPTOR ? 128 : 64;
    for (i = 0; i < kp1->total; i++) {
        // find average and standard deviation of each descriptor
        avg1[i] = 0;
        dev1[i] = 0;
        seq1 = (float*) cvGetSeqElem(desc1, i);
        for (k = 0; k < descriptor_size; k++) avg1[i] += seq1[k];
        avg1[i] /= descriptor_size;
        for (k = 0; k < descriptor_size; k++) dev1[i] +=
                (seq1[k] - avg1[i])*(seq1[k] - avg1[i]);
        dev1[i] = sqrt(dev1[i] / descriptor_size);

        // initialize best1 and best1corr
        best1[i] = -1;
        best1corr[i] = -1.;
    }
    for (j = 0; j < kp2->total; j++) {
        // find average and standard deviation of each descriptor
        avg2[j] = 0;
        dev2[j] = 0;
        seq2 = (float*) cvGetSeqElem(desc2, j);
        for (k = 0; k < descriptor_size; k++) avg2[j] += seq2[k];
        avg2[j] /= descriptor_size;
        for (k = 0; k < descriptor_size; k++) dev2[j] +=
                (seq2[k] - avg2[j])*(seq2[k] - avg2[j]);
        dev2[j] = sqrt(dev2[j] / descriptor_size);

        // initialize best2 and best2corr
        best2[j] = -1;
        best2corr[j] = -1.;
    }
    double corr;
    for (i = 0; i < kp1->total; ++i) {
        seq1 = (float*) cvGetSeqElem(desc1, i);
        for (j = 0; j < kp2->total; ++j) {
            corr = 0;
            seq2 = (float*) cvGetSeqElem(desc2, j);
            for (k = 0; k < descriptor_size; ++k)
                corr += (seq1[k] - avg1[i])*(seq2[k] - avg2[j]);
            corr /= (descriptor_size - 1) * dev1[i] * dev2[j];
            if (corr > best1corr[i]) {
                best1corr[i] = corr;
                best1[i] = j;
            }
            if (corr > best2corr[j]) {
                best2corr[j] = corr;
                best2[j] = i;
            }
        }
    }
    j = 0;
    for (i = 0; i < kp1->total; i++)
        if (best2[best1[i]] == i && best1corr[i] > CORRELATION_THRESHOLD)
            j++;
    if (j == 0) return; // no matches found
    *points1 = cvCreateMat(1, j, CV_32FC2);
    *points2 = cvCreateMat(1, j, CV_32FC2);
    CvPoint2D32f *p1, *p2;
    j = 0;
    for (i = 0; i < kp1->total; i++) {
        if (best2[best1[i]] == i && best1corr[i] > CORRELATION_THRESHOLD) {
            p1 = &((CvSURFPoint*) cvGetSeqElem(kp1, i))->pt;
            p2 = &((CvSURFPoint*) cvGetSeqElem(kp2, best1[i]))->pt;
            (*points1)->data.fl[j * 2] = p1->x;
            (*points1)->data.fl[j * 2 + 1] = p1->y;
            (*points2)->data.fl[j * 2] = p2->x;
            (*points2)->data.fl[j * 2 + 1] = p2->y;
            j++;
        }
    }
    free(best2corr);
    free(best1corr);
    free(best2);
    free(best1);
    free(avg1);
    free(avg2);
    free(dev1);
    free(dev2);   
}

void showPairs2(char * name, IplImage * img1, IplImage * img2, CvMat * points1, CvMat * points2) {
    CvSize size = cvGetSize(img1);
    IplImage * imgA = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * imgB = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * imgC = cvCreateImage(cvSize(size.width * 2, size.height), IPL_DEPTH_8U, 3);

    cvCvtColor(img1, imgA, CV_GRAY2BGR);
    cvCvtColor(img2, imgB, CV_GRAY2BGR);

    cvZero(imgC);
    cvSetImageROI(imgC, cvRect(0, 0, size.width, size.height));
    cvAdd(imgA, imgC, imgC);
    cvCvtColor(img2, imgB, CV_GRAY2BGR);
    cvSetImageROI(imgC, cvRect(size.width, 0, size.width, size.height));
    cvAdd(imgB, imgC, imgC);
    cvResetImageROI(imgC);
    CvPoint2D32f p1, p2;
    CvScalar p;

    for (int i = 0; i < points1->cols; i++) {
        p = cvGet2D(points1, 0, i);
        p1 = cvPoint2D32f(p.val[0], p.val[1]);
        p = cvGet2D(points2, 0, i);
        p2 = cvPoint2D32f(p.val[0], p.val[1]);

        CvScalar color = cvScalar(rand() % 255, rand() % 255, rand() % 255);
        cvCircle(imgC, cvPointFrom32f(p1), 2, color, -1);
        cvCircle(imgC, cvPoint((int) p2.x + size.width, (int) p2.y), 2, color, -1);
    }

    cvNamedWindow(name, 1);
    cvShowImage(name, imgC);

    cvReleaseImage(&imgA);
    cvReleaseImage(&imgB);
    cvReleaseImage(&imgC);
}

/*If you construct a fundamental matrix(a model) for the transformation
    between the two rectangles, you can further determine which correspondences
    are false (by how well they fit the model) and remove them, which I like to
    do like this :

            F = cvCreateMat(3, 3, CV_32FC1);
CvMat *status = cvCreateMat(1, points1->cols, CV_8UC1);
int fm_count = cvFindFundamentalMat(points1, points2, F,
        CV_FM_RANSAC, 1., 0.99, status);
removeOutliers(&points1, &points2, status);

where removeOutliers() is a function I wrote to clean up after
cvFindFundamentalMat() :*/

// iterates the set of putative correspondences and removes correspondences
//marked as outliers by cvFindFundamentalMat()
void removeOutliers(CvMat **points1, CvMat **points2, CvMat *status) {
    CvMat *points1_ = *points1;
    CvMat *points2_ = *points2;
    int count = 0;
    for (int i = 0; i < status->cols; i++) if (CV_MAT_ELEM(*status, unsigned
                char, 0, i)) count++;
    if (!count) { // no inliers
        *points1 = NULL;
        *points2 = NULL;
    } else {
        *points1 = cvCreateMat(1, count, CV_32FC2);
        *points2 = cvCreateMat(1, count, CV_32FC2);
        int j = 0;
        for (int i = 0; i < status->cols; i++) {
            if (CV_MAT_ELEM(*status, unsigned char, 0, i)) {
                (*points1)->data.fl[j * 2] = points1_->data.fl[i * 2];
                //p1->x
                (*points1)->data.fl[j * 2 + 1] = points1_->data.fl[i * 2 + 1];
                //p1->y
                (*points2)->data.fl[j * 2] = points2_->data.fl[i * 2];
                //p2->x
                (*points2)->data.fl[j * 2 + 1] = points2_->data.fl[i * 2 + 1];
                //p2->y
                j++;
            }
        }
    }
    cvReleaseMat(&points1_);
    cvReleaseMat(&points2_);
}

// STAR

int starTest(const IplImage * img, char * name) {
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* keypoints = 0;
    int i;

    if( !img )
        return 0;
    cvNamedWindow(name, 1 );
    IplImage * cimg = cvCreateImage( cvGetSize(img), 8, 3 );
    cvCvtColor( img, cimg, CV_GRAY2BGR );

    keypoints = cvGetStarKeypoints( img, storage, cvStarDetectorParams(45) );

    for( i = 0; i < (keypoints ? keypoints->total : 0); i++ )
    {
        CvStarKeypoint kpt = *(CvStarKeypoint*)cvGetSeqElem(keypoints, i);
        int r = kpt.size/2;
        cvCircle( cimg, kpt.pt, r, CV_RGB(0,255,0));
        cvLine( cimg, cvPoint(kpt.pt.x + r, kpt.pt.y + r),
            cvPoint(kpt.pt.x - r, kpt.pt.y - r), CV_RGB(0,255,0));
        cvLine( cimg, cvPoint(kpt.pt.x - r, kpt.pt.y + r),
            cvPoint(kpt.pt.x + r, kpt.pt.y - r), CV_RGB(0,255,0));
        cout << kpt.pt.x << ", " << kpt.pt.y << " => " << kpt.response << endl;
    }
    cvShowImage(name, cimg );
    cout << "===============" << endl;
}


// MSER

static CvScalar colors[] =
    {
        {{0,0,255}},
        {{0,128,255}},
        {{0,255,255}},
        {{0,255,0}},
        {{255,128,0}},
        {{255,255,0}},
        {{255,0,0}},
        {{255,0,255}},
        {{255,255,255}},
	{{196,255,255}},
	{{255,255,196}}
    };

    static uchar bcolors[][3] =
    {
        {0,0,255},
        {0,128,255},
        {0,255,255},
        {0,255,0},
        {255,128,0},
        {255,255,0},
        {255,0,0},
        {255,0,255},
        {255,255,255}
    };

    
int mesrTest(const IplImage * img, char * name, t_moment * &momentList, int &nMoments) {    
    double t = cvGetTickCount();
	CvSeq* contours;
	CvMemStorage* storage= cvCreateMemStorage();
        IplImage * maskRegions = cvCreateImage( cvGetSize( img ), IPL_DEPTH_8U, 1);
        IplImage * result = cvCreateImage( cvGetSize( img ), IPL_DEPTH_8U, 3);
        cvCopyImage(img, maskRegions);
        cvCvtColor(img, result, CV_GRAY2BGR);
        cvNamedWindow(name, 1);

        //cvExtractMSER(maskRegions, NULL, &contours, storage, cvMSERParams( 5, 60, 1000, .25, .2, 200, 1.01, .003, 5 ) );
        cvExtractMSER(maskRegions, NULL, &contours, storage, cvMSERParams( 5, 200, 5000, .25, .2, 200, 1.01, .003, 1 ) );

        momentList = new t_moment[contours->total];
        nMoments = contours->total;
        // draw mser with different color
	for ( int i = contours->total-1; i >= 0; i--) {
            cvZero(maskRegions);
            CvSeq* r = *(CvSeq**)cvGetSeqElem( contours, i );
            CvBox2D box = cvMinAreaRect2( r );
            //CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);
            momentList[i].points = new CvPoint[r->total];
            momentList[i].nPoints = r->total;
            for ( int j = 0; j < r->total; j++ ) {
                CvPoint* pt = CV_GET_SEQ_ELEM( CvPoint, r, j );
                cvSet2D(maskRegions, pt->y, pt->x, cvScalar(255));
                //cvSet2D(result, pt->y, pt->x, color);
                momentList[i].points[j] = cvPoint(pt->x, pt->y);
            }            
            CvMoments moments;
            CvHuMoments hu;
            cvMoments(maskRegions, &moments, 1);
            cvGetHuMoments(&moments, &hu);
            momentList[i].box = box;
            momentList[i].moments = moments;
            momentList[i].hu = hu;

            momentList[i].normHu[0] = ((momentList[i].hu.hu1 < 0)? -1: 1) * log(abs(momentList[i].hu.hu1));
            momentList[i].normHu[1] = ((momentList[i].hu.hu2 < 0)? -1: 1) * log(abs(momentList[i].hu.hu2));
            momentList[i].normHu[2] = ((momentList[i].hu.hu3 < 0)? -1: 1) * log(abs(momentList[i].hu.hu3));
            momentList[i].normHu[3] = ((momentList[i].hu.hu4 < 0)? -1: 1) * log(abs(momentList[i].hu.hu4));
            momentList[i].normHu[4] = ((momentList[i].hu.hu5 < 0)? -1: 1) * log(abs(momentList[i].hu.hu5));
            momentList[i].normHu[5] = ((momentList[i].hu.hu6 < 0)? -1: 1) * log(abs(momentList[i].hu.hu6));
            momentList[i].normHu[6] = ((momentList[i].hu.hu7 < 0)? -1: 1) * log(abs(momentList[i].hu.hu7));
            
            cvEllipseBox( result, box, colors[10], 1 );            
            CvPoint2D32f pts[4];
            cvBoxPoints(box, pts);
            //cvRectangle(result, cvPointFrom32f(pts[0]), cvPointFrom32f(pts[2]), colors[10]);
            for (int i = 0; i < 4; i++) {
                if (i != 3)
                    cvLine(result, cvPointFrom32f(pts[i]), cvPointFrom32f(pts[i + 1]), cvScalar(0, 255, 0), 1);
                else cvLine(result, cvPointFrom32f(pts[i]), cvPointFrom32f(pts[0]), cvScalar(0, 255, 0), 1);
            }
            cvLine(result, cvPointFrom32f(box.center), cvPoint((pts[0].x + pts[1].x) / 2, (pts[0].y + pts[1].y) / 2), cvScalar(0, 255, 0), 1);
            cvCircle(result, cvPointFrom32f(box.center), 2, cvScalar(0, 255, 0));
	}

        t = cvGetTickCount() - t;
	printf( "MSER extracted %d in %g ms.\n", contours->total, t/((double)cvGetTickFrequency()*1000.) );

        cvShowImage(name, result);
        //cvWaitKey(0);

        cvReleaseImage(&result);
        cvReleaseImage(&maskRegions);
}

double calcCorr(t_moment mmt1, t_moment mmt2, int method) {
    if (method == 1) {
        double sum = 0;
        for (int i = 0; i < 7; i++) {
            double val = abs((1 / mmt1.normHu[i]) - (1 / mmt2.normHu[i]));
            if (isinf(val)) {
                cout << "val = " << val << ", " << mmt1.normHu[i] << ", " << mmt2.normHu[i] << ", " << mmt1.hu.hu7 << ", " << mmt2.hu.hu7 << endl;
                continue;
            }
            sum += val;
        }
        return sum;
    } else if (method == 2) {
        double sum = 0;
        for (int i = 0; i < 7; i++) {
            double val = abs(mmt1.normHu[i] - mmt2.normHu[i]);
            if (isinf(val)) {
                cout << "val = " << val << ", " << mmt1.normHu[i] << ", " << mmt2.normHu[i] << ", " << mmt1.hu.hu7 << ", " << mmt2.hu.hu7 << endl;
                continue;
            }
            sum += val;
        }
        return sum;
    } else if (method == 3) {
        double sum = 0;
        for (int i = 0; i < 7; i++) {
            double val = abs(mmt1.normHu[i] - mmt2.normHu[i]) / abs(mmt1.normHu[i]);
            if (isinf(val)) {
                cout << "val = " << val << ", " << mmt1.normHu[i] << ", " << mmt2.normHu[i] << ", " << mmt1.hu.hu7 << ", " << mmt2.hu.hu7 << endl;
                continue;
            }
            sum += val;
        }
        return sum;
    }
    return 0;
}

double calcCCorr(IplImage * img1, IplImage * img2, t_moment mmt1, t_moment mmt2, bool show) {
    CvPoint2D32f pts1[4], pts2[4], destPoints[4];
    cvBoxPoints(mmt1.box, pts1);
    cvBoxPoints(mmt2.box, pts2);

    CvSize newSize = cvSize(min(mmt1.box.size.width, mmt2.box.size.width), min(mmt1.box.size.height, mmt2.box.size.height));
    destPoints[0] = cvPoint2D32f(0, 0);
    destPoints[1] = cvPoint2D32f(newSize.width, 0);
    destPoints[2] = cvPoint2D32f(newSize.width, newSize.height);
    destPoints[3] = cvPoint2D32f(0, newSize.height);

    CvMat * mat1 = cvCreateMat(3, 3, CV_32FC1);
    CvMat * mat2 = cvCreateMat(3, 3, CV_32FC1);

    cvGetPerspectiveTransform(pts1, destPoints, mat1);
    cvGetPerspectiveTransform(pts2, destPoints, mat2);

    IplImage * dest1 = cvCreateImage(newSize, IPL_DEPTH_8U, 1);
    IplImage * dest2 = cvCreateImage(newSize, IPL_DEPTH_8U, 1);

    cvWarpPerspective(img1, dest1, mat1, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
    cvWarpPerspective(img2, dest2, mat2, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, cvScalarAll(0));

    if (show) {
        cvNamedWindow("Dest1", 1);
        cvNamedWindow("Dest2", 1);

        cvShowImage("Dest1", dest1);
        cvShowImage("Dest2", dest2);
    }

    CvScalar mean1, std1, mean2, std2;
    cvAvgSdv(dest1, &mean1, &std1);
    cvAvgSdv(dest2, &mean2, &std2);

    double corr = 0;
    for (int i = 0; i < newSize.height; i++) {
        for (int j = 0; j < newSize.width; j++) {
            corr += (cvGetReal2D(dest1, i, j) - mean1.val[0]) * (cvGetReal2D(dest2, i, j) - mean2.val[0]);
        }
    }
    corr /= (newSize.width * newSize.height - 1) * std1.val[0] * std2.val[0];

//    cvWaitKey(0);

    cvReleaseImage(&dest1);
    cvReleaseImage(&dest2);

    cvReleaseMat(&mat1);
    cvReleaseMat(&mat2);

    return corr;
}


void matchMserByMoments(IplImage * img1, IplImage * img2, t_moment * momentList1, t_moment * momentList2, int nMoment1, int nMoment2, char * name, vector<t_moment *> &regionPairs) {
    double t = cvGetTickCount();
    IplImage * mask1 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
    IplImage * mask2 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
    IplImage * masked1 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
    IplImage * masked2 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);

    cvZero(mask1);
    cvZero(mask2);
    cvZero(masked1);
    cvZero(masked2);
    for (int i = 0; i < nMoment1; i++) {
        for (int j = 0; j < momentList1[i].nPoints; j++) {
            cvSet2D(mask1, momentList1[i].points[j].y, momentList1[i].points[j].x, cvScalar(255));
        }
    }
    for (int i = 0; i < nMoment2; i++) {
        for (int j = 0; j < momentList2[i].nPoints; j++) {
            cvSet2D(mask2, momentList2[i].points[j].y, momentList2[i].points[j].x, cvScalar(255));
        }
    }

    cvCopy(img1, masked1, NULL);
    cvCopy(img2, masked2, NULL);

    CvMat * matches = cvCreateMat(nMoment1, nMoment2, CV_32FC1);
    for (int i = 0; i < nMoment1; i++) {
        for (int j = 0; j < nMoment2; j++) {            
            double corr = calcCCorr(masked1, masked2, momentList1[i], momentList2[j], false);
            cvSetReal2D(matches, i, j, corr);
        }
    }

    CvSize size = cvGetSize(img1);
    IplImage * result = cvCreateImage(cvSize(img1->width * 2, img1->height), IPL_DEPTH_8U, 3);
    cvZero(result);
    cvSetImageROI(result, cvRect(0, 0, size.width, size.height));    
    cvCvtColor(img1, result, CV_GRAY2BGR);
    cvSetImageROI(result, cvRect(size.width, 0, size.width, size.height));
    cvCvtColor(img2, result, CV_GRAY2BGR);
    cvResetImageROI(result);

    cvNamedWindow(name, 1);    

    for (int k = 0; k < min(nMoment1, nMoment2); k++) {
        double maxVal = DBL_MIN;
        CvPoint pIndex = cvPoint(-1, -1);
        for (int i = 0; i < nMoment1; i++) {
            for (int j = 0; j < nMoment2; j++) {
                if (cvGetReal2D(matches, i, j)  > maxVal) {
                    maxVal = cvGetReal2D(matches, i, j);
                    pIndex = cvPoint(i, j);
                }
            }
        }
        if (maxVal < CORRELATION_THRESHOLD_MESR)
            break;
        for (int i = 0; i < nMoment1; i++)
            cvSetReal2D(matches, i, pIndex.y, DBL_MIN);
        for (int i = 0; i < nMoment2; i++)
            cvSetReal2D(matches, pIndex.x, i, DBL_MIN);
        //cvSetReal2D(matches, pIndex.x, pIndex.y, DBL_MIN);
        t_moment mmt1 = momentList1[pIndex.x];
        t_moment mmt2 = momentList2[pIndex.y];          

        CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);
        for (int i = 0; i < mmt1.nPoints; i++) {
            cvSet2D(result, mmt1.points[i].y, mmt1.points[i].x, color);
        }
        for (int i = 0; i < mmt2.nPoints; i++) {
            cvSet2D(result, mmt2.points[i].y, mmt2.points[i].x + size.width, color);
        }

        t_moment * pair = new t_moment[2];
        pair[0] = mmt1;
        pair[1] = mmt2;
        regionPairs.push_back(pair);

        //calcCCorr(masked1, masked2, mmt1, mmt2, true);

        cout << maxVal << endl;

        /*cvShowImage(name, result);
        if (cvWaitKey(0) == 27)
            exit(0);        //*/
                
    }
    t = cvGetTickCount() - t;
    printf( "MSER matched %d in %g ms.\n", regionPairs.size(), t/((double)cvGetTickFrequency()*1000.) );

    cvShowImage(name, result);
 
    cvReleaseImage(&result);
    cvReleaseImage(&mask1);
    cvReleaseImage(&mask2);
    cvReleaseImage(&masked1);
    cvReleaseImage(&masked2);
}

void cleanMatches(IplImage * img1, IplImage * img2, vector<t_moment *> &regionPairs, char * name, CvPoint2D32f * &points1, CvPoint2D32f * &points2, int &nFeat) {

    CvSize size = cvGetSize(img1);
    IplImage * result = cvCreateImage(cvSize(img1->width * 2, img1->height), IPL_DEPTH_8U, 3);    

    cvNamedWindow(name, 1);
    cvZero(result);
    cvSetImageROI(result, cvRect(0, 0, size.width, size.height));
    cvCvtColor(img1, result, CV_GRAY2BGR);
    cvSetImageROI(result, cvRect(size.width, 0, size.width, size.height));
    cvCvtColor(img2, result, CV_GRAY2BGR);
    cvResetImageROI(result);

    points1 = new CvPoint2D32f[regionPairs.size() * 5];
    points2 = new CvPoint2D32f[regionPairs.size() * 5];
    nFeat = 0;
    for (int i = 0; i < regionPairs.size(); i++) {
        CvBox2D box1 = regionPairs.at(i)[0].box;
        CvBox2D box2 = regionPairs.at(i)[1].box;
        CvPoint2D32f pts1[4], pts2[4];
        cvBoxPoints(box1, pts1);
        cvBoxPoints(box2, pts2);
        if ((box1.center.x >= 0) && (box1.center.y >= 0) &&
            (box2.center.x >= 0) && (box2.center.y >= 0) &&
            (box1.center.x < img1->width) && (box1.center.y < img1->height) &&
            (box2.center.x < img2->width) && (box2.center.y < img2->height)) {

            points1[nFeat] = box1.center;
            points2[nFeat] = box2.center;
            nFeat++;
        }
        for (int j = 0; j < 4; j++) {
            if ((pts1[j].x >= 0) && (pts1[j].y >= 0) &&
            (pts2[j].x >= 0) && (pts2[j].y >= 0) &&
            (pts1[j].x < img1->width) && (pts1[j].y < img1->height) &&
            (pts2[j].x < img2->width) && (pts2[j].y < img2->height)) {
                points1[nFeat] = pts1[j];
                points2[nFeat] = pts2[j];
                nFeat++;
            }
        }
    }

    for (int i = 0; i < nFeat; i++) {
        //CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);
        CvScalar color = cvScalar(0, 0, 255);
        cvCircle(result, cvPointFrom32f(points1[i]), 2, color);
        cvCircle(result, cvPoint(points2[i].x + size.width, points2[i].y), 2, color);
    }

    //CImageRegistration registration(size);
    //registration.cleanFeat(points1, points2, nFeat);

    for (int i = 0; i < nFeat; i++) {
        CvScalar color = cvScalar(0, 255, 0);
        cvCircle(result, cvPointFrom32f(points1[i]), 2, color);
        cvCircle(result, cvPoint(points2[i].x + size.width, points2[i].y), 2, color);
    }

    cvShowImage(name, result);
    /*if (cvWaitKey(0) == 27)
        exit(0);        //*/

    cvReleaseImage(&result);
}