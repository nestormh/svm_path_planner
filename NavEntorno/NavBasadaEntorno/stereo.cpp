#include "CRealMatches.h"

void CRealMatches::test3D() {
    int nPairs = pairs.size();
    CvMat * p1 = cvCreateMat(1, nPairs, CV_32FC2);
    CvMat * p2 = cvCreateMat(1, nPairs, CV_32FC2);
    CvMat* H1 = cvCreateMat(3, 3, CV_64FC1);
    CvMat* H2 = cvCreateMat(3, 3, CV_64FC1);
    CvMat * F = cvCreateMat(3, 3, CV_32FC1);
    CvMat * epilines = cvCreateMat(3, nPairs, CV_32FC1);
    CvMat *statusM = cvCreateMat(1, nPairs, CV_8UC1);

    for (int i = 0; i < nPairs; i++) {
        cvSet2D(p1, 0, i, cvScalar(pairs.at(i).p1.x, pairs.at(i).p1.y));
        cvSet2D(p2, 0, i, cvScalar(pairs.at(i).p2.x, pairs.at(i).p2.y));
    }

    /*cvFindFundamentalMat(p1, p2, F, CV_FM_RANSAC, 3., 0.99, statusM);

    cvComputeCorrespondEpilines(p1, 1, F, epilines);

    cvStereoRectifyUncalibrated(p1, p2, F, size, H1, H2, 5);*/

    // NOTA: R_rect = M*H*M
    // M = camera matrix
    // Falta por saber coeffs y M

    CvMat* mx1 = cvCreateMat(size.height, size.width, CV_32F);
    CvMat* my1 = cvCreateMat(size.height, size.width, CV_32F);
    CvMat* mx2 = cvCreateMat(size.height, size.width, CV_32F);
    CvMat* my2 = cvCreateMat(size.height, size.width, CV_32F);
    CvMat* img1r = cvCreateMat(size.height, size.width, CV_8U);
    CvMat* img2r = cvCreateMat( size.height, size.width, CV_8U);
    CvMat* disp = cvCreateMat(size.height, size.width, CV_16S);
    CvMat* vdisp = cvCreateMat(size.height, size.width, CV_8U);
    CvMat* pair;
    double R1[3][3], R2[3][3], P1[3][4], P2[3][4];
    CvMat _R1 = cvMat(3, 3, CV_64F, R1);
    CvMat _R2 = cvMat(3, 3, CV_64F, R2);

    double iM[3][3];
    CvMat _iM = cvMat(3, 3, CV_64F, iM);
    
    cvFindFundamentalMat(p1, p2, F);
    cvStereoRectifyUncalibrated(p1, p2, F, size, H1, H2, 3);

    cvInvert(M1, &_iM);
    cvMatMul(H1, M1, &_R1);
    cvMatMul(&_iM, &_R1, &_R1);
    cvInvert(M2, &_iM);
    cvMatMul(H2, M2, &_R2);
    cvMatMul(&_iM, &_R2, &_R2);
    //Precompute map for cvRemap()
    cvInitUndistortRectifyMap(M1, D1, &_R1, M1, mx1, my1);
    cvInitUndistortRectifyMap(M2, D1, &_R2, M2, mx2, my2);

    IplImage * distort1 = cvCreateImage(size, IPL_DEPTH_8U, 1);
    IplImage * distort2 = cvCreateImage(size, IPL_DEPTH_8U, 1);

    cvRemap( img1, distort1, mx1, my1 );
    cvRemap( img2, distort2, mx2, my2 );

    cvNamedWindow("distort1", 1);
    cvNamedWindow("distort2", 1);
    cvShowImage("distort1", distort1);
    cvShowImage("distort2", distort2);

    CvPoint2D32f point1, point2;
    double max = DBL_MIN;
    vector<t_Pair3D> pairs3D;
    for (int i = 0; i < pairs.size(); i++) {        
        point1 = cvPoint2D32f(cvGetReal2D(mx1, pairs.at(i).p1.y, pairs.at(i).p1.x), cvGetReal2D(my1, pairs.at(i).p1.y, pairs.at(i).p1.x));
        point2 = cvPoint2D32f(cvGetReal2D(mx2, pairs.at(i).p2.y, pairs.at(i).p2.x), cvGetReal2D(my2, pairs.at(i).p2.y, pairs.at(i).p2.x));

        t_Pair3D pair;
        pair.p1 = cvPoint3D32f(pairs.at(i).p1.x, pairs.at(i).p1.y, abs(point1.x - point2.x));
        pair.p2 = cvPoint3D32f(pairs.at(i).p2.x, pairs.at(i).p2.y, abs(point1.x - point2.x));

        pairs3D.push_back(pair);

        if (max < abs(point1.x - point2.x))
            max = abs(point1.x - point2.x);
    }

    IplImage * xz = cvCreateImage(cvSize(size.width, size.width), IPL_DEPTH_8U, 3);
    cvZero(xz);
    for (int i = 0; i < pairs3D.size(); i++) {
        cvCircle(xz, cvPoint((int)pairs3D.at(i).p1.x, (int)(pairs3D.at(i).p1.z * size.width / max)), 2, cvScalar(0, 255, 0), -1);
    }

    cvNamedWindow("xz", 1);
    cvShowImage("xz", xz);

}

void CRealMatches::calibrateCameras() {
    char * localPath = "/home/neztol/doctorado/samplesOpenCV/c/";
    char fullPath[1024];
    sprintf(fullPath, "%sstereo_calib.txt", localPath);
    FILE* f = fopen(fullPath, "rt");    
    if (!f) {
        fprintf(stderr, "can not open file %s\n", fullPath);
        return;
    }

    int nx = 9;
    int ny = 6;
    int useUncalibrated = 2;
    int displayCorners = 0;
    int showUndistorted = 1;
    bool isVerticalStereo = false;//OpenCV can handle left-right
                                      //or up-down camera arrangements
    const int maxScale = 1;
    const float squareSize = 1.f; //Set this to your actual square size
    int i, j, lr, nframes, n = nx*ny, N = 0;
    vector<string> imageNames[2];
    vector<CvPoint3D32f> objectPoints;
    vector<CvPoint2D32f> points[2];
    vector<int> npoints;
    vector<uchar> active[2];
    vector<CvPoint2D32f> temp(n);
    CvSize imageSize = {0,0};
    // ARRAY AND VECTOR STORAGE:
    double M1[3][3], M2[3][3], D1[5], D2[5];
    double R[3][3], T[3], E[3][3], F[3][3];
    CvMat _M1 = cvMat(3, 3, CV_64F, M1 );
    CvMat _M2 = cvMat(3, 3, CV_64F, M2 );
    CvMat _D1 = cvMat(1, 5, CV_64F, D1 );
    CvMat _D2 = cvMat(1, 5, CV_64F, D2 );
    CvMat _R = cvMat(3, 3, CV_64F, R );
    CvMat _T = cvMat(3, 1, CV_64F, T );
    CvMat _E = cvMat(3, 3, CV_64F, E );
    CvMat _F = cvMat(3, 3, CV_64F, F );
    if( displayCorners )
        cvNamedWindow( "corners", 1 );

    for (int i = 0;; i++) {
        char buf[1024];
        int count = 0, result = 0;
        int lr = i % 2;

        vector<CvPoint2D32f>& pts = points[lr];

        if (!fgets(buf, sizeof (buf) - 3, f))
            break;
        size_t len = strlen(buf);

        while (len > 0 && isspace(buf[len - 1]))
            buf[--len] = '\0';
        if (buf[0] == '#')
            continue;

        sprintf(fullPath, "%s%s", localPath, buf);
        IplImage* img = cvLoadImage(fullPath, 0);
        if (!img)
            break;
        imageSize = cvGetSize(img);
        imageNames[lr].push_back(fullPath);
        //FIND CHESSBOARDS AND CORNERS THEREIN:
        for (int s = 1; s <= maxScale; s++) {
            IplImage* timg = img;
            if (s > 1) {
                timg = cvCreateImage(cvSize(img->width*s, img->height * s),
                        img->depth, img->nChannels);
                cvResize(img, timg, CV_INTER_CUBIC);
            }
            result = cvFindChessboardCorners(timg, cvSize(nx, ny),
                    &temp[0], &count,
                    CV_CALIB_CB_ADAPTIVE_THRESH |
                    CV_CALIB_CB_NORMALIZE_IMAGE);
            if (timg != img)
                cvReleaseImage(&timg);
            if (result || s == maxScale)
                for (j = 0; j < count; j++) {
                    temp[j].x /= s;
                    temp[j].y /= s;
                }
            if (result)
                break;
        }

        if (displayCorners) {
            printf("%s\n", fullPath);
            IplImage* cimg = cvCreateImage(imageSize, 8, 3);
            cvCvtColor(img, cimg, CV_GRAY2BGR);
            cvDrawChessboardCorners(cimg, cvSize(nx, ny), &temp[0],
                    count, result);
            cvShowImage("corners", cimg);
            cvReleaseImage(&cimg);
            int c = cvWaitKey(1000);
            if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
                exit(-1);
        } else
            putchar('.');
        N = pts.size();
        pts.resize(N + n, cvPoint2D32f(0, 0));
        active[lr].push_back((uchar) result);
        //assert( result != 0 );
        if (result) {
            //Calibration will suffer without subpixel interpolation
            cvFindCornerSubPix(img, &temp[0], count,
                    cvSize(11, 11), cvSize(-1, -1),
                    cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS,
                    30, 0.01));
            copy(temp.begin(), temp.end(), pts.begin() + N);
        }
        cvReleaseImage(&img);
    }
    fclose(f);
    printf("\n");

    // HARVEST CHESSBOARD 3D OBJECT POINT LIST:
    nframes = active[0].size(); //Number of good chessboads found
    objectPoints.resize(nframes * n);
    for (i = 0; i < ny; i++)
        for (j = 0; j < nx; j++)
            objectPoints[i * nx + j] =
                cvPoint3D32f(i * squareSize, j * squareSize, 0);
    for (i = 1; i < nframes; i++)
        copy(objectPoints.begin(), objectPoints.begin() + n,
            objectPoints.begin() + i * n);
    npoints.resize(nframes, n);
    N = nframes*n;
    CvMat _objectPoints = cvMat(1, N, CV_32FC3, &objectPoints[0]);
    CvMat _imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0]);
    CvMat _imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0]);
    CvMat _npoints = cvMat(1, npoints.size(), CV_32S, &npoints[0]);
    cvSetIdentity(&_M1);
    cvSetIdentity(&_M2);
    cvZero(&_D1);
    cvZero(&_D2);

    // CALIBRATE THE STEREO CAMERAS
    printf("Running stereo calibration ...");
    fflush(stdout);
    cvStereoCalibrate(&_objectPoints, &_imagePoints1,
            &_imagePoints2, &_npoints,
            &_M1, &_D1, &_M2, &_D2,
            imageSize, &_R, &_T, &_E, &_F,
            cvTermCriteria(CV_TERMCRIT_ITER +
            CV_TERMCRIT_EPS, 100, 1e-5),
            CV_CALIB_FIX_ASPECT_RATIO +
            CV_CALIB_ZERO_TANGENT_DIST +
            CV_CALIB_SAME_FOCAL_LENGTH);
    printf(" done\n");
    // CALIBRATION QUALITY CHECK
    // because the output fundamental matrix implicitly
    // includes all the output information,
    // we can check the quality of calibration using the
    // epipolar geometry constraint: m2^t*F*m1=0
    vector<CvPoint3D32f> lines[2];
    points[0].resize(N);
    points[1].resize(N);
    _imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0]);
    _imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0]);
    lines[0].resize(N);
    lines[1].resize(N);
    CvMat _L1 = cvMat(1, N, CV_32FC3, &lines[0][0]);
    CvMat _L2 = cvMat(1, N, CV_32FC3, &lines[1][0]);
    //Always work in undistorted space
    cvUndistortPoints( &_imagePoints1, &_imagePoints1,
        &_M1, &_D1, 0, &_M1 );
    cvUndistortPoints( &_imagePoints2, &_imagePoints2,
        &_M2, &_D2, 0, &_M2 );
    cvComputeCorrespondEpilines( &_imagePoints1, 1, &_F, &_L1 );
    cvComputeCorrespondEpilines( &_imagePoints2, 2, &_F, &_L2 );
    double avgErr = 0;
    for( i = 0; i < N; i++ )
    {
        double err = fabs(points[0][i].x*lines[1][i].x +
            points[0][i].y*lines[1][i].y + lines[1][i].z)
            + fabs(points[1][i].x*lines[0][i].x +
            points[1][i].y*lines[0][i].y + lines[0][i].z);
        avgErr += err;
    }
    printf( "avg err = %g\n", avgErr/(nframes*n) );

    cvCopy(&_M1, this->M1);
    cvCopy(&_M2, this->M2);
    cvCopy(&_D1, this->D1);
    cvCopy(&_D2, this->D2);
}

void CRealMatches::test3D_2() {
    int * distances = new int[pairs.size()];
    double minDist = DBL_MAX;
    for (int i = 0; i < pairs.size(); i++) {
        double dist = pairs.at(i).p2.x - pairs.at(i).p1.x;
        distances[i] = dist;
        if (abs(minDist) > abs(dist)) {
            minDist = dist;
        }
    }

    for (int i = 0; i < pairs.size(); i++) {
        distances[i] -= minDist;
    }

    IplImage * tmpImg1 = cvCreateImage(size, IPL_DEPTH_8U, 3);

    cvCvtColor(img1, tmpImg1, CV_GRAY2BGR);
    cvZero(tmpImg1);

    int pos = 0;
    for (vector<t_Pair>::iterator it = pairs.begin(); it != pairs.end(); it++, pos++) {
        CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);
        cvLine(tmpImg1, cvPointFrom32f(it->p1), cvPoint(it->p1.x + distances[pos], it->p1.y), color);
        //cvCircle(tmpImg1, cvPointFrom32f(it->p1), 2, color, -1);
    }

    cvNamedWindow("Distances", 1);
    cvShowImage("Distances", tmpImg1);

    cvReleaseImage(&tmpImg1);
    delete distances;
}