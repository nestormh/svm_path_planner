/* 
 * File:   SurfGPU.cpp
 * Author: nestor
 * 
 * Created on 25 de junio de 2010, 12:39
 */

#include "SurfGPU.h"
#include "CRealMatches.h"

SurfGPU::SurfGPU() {
    config.threshold=0.0001;
}

SurfGPU::SurfGPU(const SurfGPU& orig) {
}

SurfGPU::~SurfGPU() {
}

void SurfGPU::testSurfGPU(cv::Mat img, vector<KeyPoint> &points, vector<float> &descriptors, GpuSurfConfiguration config) {    
    detector.buildIntegralImage(img);
    detector.detectKeypoints();
    detector.findOrientationFast();
    detector.computeDescriptors();
	          
    detector.getKeypoints(points);
    detector.getDescriptors(descriptors);
}

inline bool SurfGPU::isLastBitSet(const int * f) {
    return (*f & 0x1);
}

inline bool SurfGPU::isLastBitSet(const float & f) {
    return isLastBitSet((const int*)&f);
}

inline CvPoint2D32f * SurfGPU::getSquare(KeyPoint k) {
        CvPoint2D32f pt[4];
        float sz = sqrt(k.size * k.size + k.size * k.size);
        float ang = k.angle - CV_PI / 4;
        float st = sz * sin(ang);
        float ct = sz * cos(ang);

        pt[0] = cvPoint2D32f(k.pt.x + ct, k.pt.y + st);

        ang = k.angle + CV_PI / 4;
        st = sz * sin(ang);
        ct = sz * cos(ang);
        pt[1] = cvPoint2D32f(k.pt.x + ct, k.pt.y + st);

        ang = k.angle + 3 * CV_PI / 4;
        st = sz * sin(ang);
        ct = sz * cos(ang);
        pt[2] = cvPoint2D32f(k.pt.x + ct, k.pt.y + st);

        ang = k.angle + 5 * CV_PI / 4;
        st = sz * sin(ang);
        ct = sz * cos(ang);
        pt[3] = cvPoint2D32f(k.pt.x + ct, k.pt.y + st);

        return pt;
}
void SurfGPU::drawKeypoints(std::vector<cv::KeyPoint> const & keypoints, IplImage * imgGrayscale, string name) {

    IplImage * imgColor = cvCreateImage(cvGetSize(imgGrayscale), IPL_DEPTH_8U, 3);
    cvCvtColor(imgGrayscale, imgColor, CV_GRAY2BGR);

    vector<KeyPoint>::const_iterator k = keypoints.begin();
    KeyPoint k2;    
    Scalar red(255, 0, 0);
    Scalar blue(0, 0, 255);

    for (; k != keypoints.end(); k++) {
        Scalar * choice = NULL;
        if (isLastBitSet(k->response))
            choice = &red;
        else
            choice = &blue;

        Point2f dir(k->pt);
        float st = k->size * sin(k->angle);
        float ct = k->size * cos(k->angle);
        dir.x += ct;
        dir.y += st;
        cvCircle(imgColor, k->pt, (int) k->size, *choice, 1, CV_AA);
        cvLine(imgColor, k->pt, dir, *choice, 1, CV_AA);
        //cvCircle(imgColor, k->pt, 2, cvScalar(0, 0, 255), -1);

        CvPoint2D32f * pt = getSquare((KeyPoint)(*k));

        cvLine(imgColor, cvPointFrom32f(pt[0]), cvPointFrom32f(pt[1]), *choice, 1, CV_AA);
        cvLine(imgColor, cvPointFrom32f(pt[1]), cvPointFrom32f(pt[2]), *choice, 1, CV_AA);
        cvLine(imgColor, cvPointFrom32f(pt[2]), cvPointFrom32f(pt[3]), *choice, 1, CV_AA);
        cvLine(imgColor, cvPointFrom32f(pt[3]), cvPointFrom32f(pt[0]), *choice, 1, CV_AA);
    }

    cvShowImage(name.c_str(), imgColor);

    cvReleaseImage(&imgColor);
}

void SurfGPU::drawPairs(vector<t_SURF_Pair> const & pairs, IplImage * imgGrayscale1, IplImage * imgGrayscale2) {

    IplImage * imgColor1 = cvCreateImage(cvGetSize(imgGrayscale1), IPL_DEPTH_8U, 3);
    IplImage * imgColor2 = cvCreateImage(cvGetSize(imgGrayscale2), IPL_DEPTH_8U, 3);
    cvCvtColor(imgGrayscale1, imgColor1, CV_GRAY2BGR);
    cvCvtColor(imgGrayscale2, imgColor2, CV_GRAY2BGR);

    for (vector<t_SURF_Pair>::const_iterator pair = pairs.begin(); pair != pairs.end(); pair++) {
        CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);

        Point2f dir1(pair->kp1.pt);
        float st = pair->kp1.size * sin(pair->kp1.angle);
        float ct = pair->kp1.size * cos(pair->kp1.angle);
        dir1.x += ct;
        dir1.y += st;
        //cvCircle(imgColor1, pair->kp1.pt, (int) pair->kp1.size, color, 1, CV_AA);
        //cvLine(imgColor1, pair->kp1.pt, dir1, color, 1, CV_AA);//*/
        //if (isLastBitSet(pair->kp2.response) == isLastBitSet(pair->kp1.response))
            cvCircle(imgColor1, pair->kp1.pt, 2, color, -1, CV_AA);

        Point2f dir2(pair->kp2.pt);
        st = pair->kp2.size * sin(pair->kp2.angle);
        ct = pair->kp2.size * cos(pair->kp2.angle);
        dir2.x += ct;
        dir2.y += st;
        //cvCircle(imgColor2, pair->kp2.pt, (int) pair->kp2.size, color, 1, CV_AA);
        //cvLine(imgColor2, pair->kp2.pt, dir2, color, 1, CV_AA);
        //if (isLastBitSet(pair->kp2.response) == isLastBitSet(pair->kp1.response))
            cvCircle(imgColor2, pair->kp2.pt, 2, color, -1, CV_AA);
    }

    cvShowImage("surfMatch1", imgColor1);
    cvShowImage("surfMatch2", imgColor2);    

    cvReleaseImage(&imgColor1);
    cvReleaseImage(&imgColor2);
}

void SurfGPU::bruteMatch(vector<KeyPoint> points1, vector<KeyPoint> points2, vector<float> desc1, vector<float> desc2, vector<t_SURF_Pair> &pairs) {

    vector<t_Point> tmpPoints1;
    vector<t_Point> tmpPoints2;
    vector<int> matches;

    for (int i = 0; i < points1.size(); i += 1) {
        t_Point p;
        p.x = points1.at(i).pt.x;
        p.y = points1.at(i).pt.y;
        p.response = isLastBitSet(points1.at(i).response);

        tmpPoints1.push_back(p);
    }

    for (int i = 0; i < points2.size(); i += 1) {
        t_Point p;
        p.x = points2.at(i).pt.x;
        p.y = points2.at(i).pt.y;
        p.response = isLastBitSet(points2.at(i).response);

        tmpPoints2.push_back(p);
    }

    clock_t myTime = clock();
    bruteMatchParallel(tmpPoints1, tmpPoints2, desc1, desc2, matches);
    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo match 1 = " << time << endl;

    cout << endl;

    for (int i = 0; i < matches.size(); i++) {
        if ((matches.at(i) != -1) && (matches.at(i) < points2.size())) {
            t_SURF_Pair pair;
            pair.kp1 = points1.at(i);
            pair.kp2 = points2.at(matches.at(i));
            pairs.push_back(pair);
        }//*/
    }

    /*myTime = clock();
    bruteMatchParallel2(tmpPoints1, tmpPoints2, desc1, desc2, matches);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo match 2 = " << time << endl;


    for (int i = 0; i < matches.size(); i++) {
        if (matches.at(i) != -1) {
            t_SURF_Pair pair;
            pair.kp1 = points1.at(i);
            pair.kp2 = points2.at(matches.at(i));
            pairs.push_back(pair);
        }        
    }*/
}

void SurfGPU::bruteMatchSequential(vector<KeyPoint> points1, vector<KeyPoint> points2, vector<float> desc1, vector<float> desc2, vector<t_SURF_Pair> &pairs) {
    float* avg1 = (float*) malloc(sizeof (float) * points1.size());
    float* avg2 = (float*) malloc(sizeof (float) * points2.size());
    float* dev1 = (float*) malloc(sizeof (float) * points1.size());
    float* dev2 = (float*) malloc(sizeof (float) * points2.size());

    int* best1 = (int*) malloc(sizeof (int) * points1.size());
    int* best2 = (int*) malloc(sizeof (int) * points2.size());

    float* best1corr = (float*) malloc(sizeof (float) * points1.size());
    float* best2corr = (float*) malloc(sizeof (float) * points2.size());
    
    int descriptor_size = 64;
    for (int i = 0; i < points1.size(); i++) {
        // find average and standard deviation of each descriptor
        avg1[i] = 0;
        dev1[i] = 0;

        for (int k = i * descriptor_size; k < (i * descriptor_size) + descriptor_size; k++) {
            avg1[i] += desc1.at(k);
        }
        avg1[i] /= descriptor_size;
        for (int k = i * descriptor_size; k < (i * descriptor_size) + descriptor_size; k++) {
            dev1[i] += (desc1.at(k) - avg1[i]) * (desc1.at(k) - avg1[i]);
        }
        dev1[i] = sqrt(dev1[i] / descriptor_size);        

        // initialize best1 and best1corr
        best1[i] = -1;
        best1corr[i] = -1.;
    }    
    for (int i = 0; i < points2.size(); i++) {
        // find average and standard deviation of each descriptor
        avg2[i] = 0;
        dev2[i] = 0;

        for (int k = i * descriptor_size; k < (i * descriptor_size) + descriptor_size; k++) {
            avg2[i] += desc2.at(k);
        }
        avg2[i] /= descriptor_size;
        for (int k = i * descriptor_size; k < (i * descriptor_size) + descriptor_size; k++) {
            dev2[i] += (desc2.at(k) - avg2[i]) * (desc2.at(k) - avg2[i]);
        }
        dev2[i] = sqrt(dev2[i] / descriptor_size);

        // initialize best2 and best2corr
        best2[i] = -1;
        best2corr[i] = -1.;
    }

    float corr;
    for (int i = 0; i < points1.size(); i++) {
        vector<float> descriptor1;
        for (int k = i * descriptor_size; k < (i * descriptor_size) + descriptor_size; k++)
            descriptor1.push_back(desc1.at(k));
        for (int j = 0; j < points2.size(); j++) {
            vector<float> descriptor2;
            for (int k = j * descriptor_size; k < (j * descriptor_size) + descriptor_size; k++)
                descriptor2.push_back(desc2.at(k));

            corr = 0;
            for (int k = 0; k < descriptor_size; k++)
                corr += (descriptor1.at(k) - avg1[i]) * (descriptor2.at(k) - avg2[j]);
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


    float CORRELATION_THRESHOLD = 0.75;
    for (int i = 0; i < points1.size(); i++) {
        if (best2[best1[i]] == i && best1corr[i] > CORRELATION_THRESHOLD) {
            t_SURF_Pair pair;
            pair.kp1 = points1.at(i);
            pair.kp2 = points2.at(best1[i]);
            pairs.push_back(pair);
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

inline void SurfGPU::removeOutliers(CvMat **points1, CvMat **points2, CvMat *status) {
    CvMat *points1_ = *points1;
    CvMat *points2_ = *points2;
    int count = 0;
    for (int i = 0; i < status->cols; i++)
        if (cvGetReal2D(status, 0, i))
            count++;
    if (!count) { // no inliers
        *points1 = NULL;
        *points2 = NULL;
    } else {
        *points1 = cvCreateMat(1, count, CV_32FC2);
        *points2 = cvCreateMat(1, count, CV_32FC2);
        int j = 0;
        for (int i = 0; i < status->cols; i++) {
            if (cvGetReal2D(status, 0, i)) {
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

inline void SurfGPU::cleanDistances(IplImage * img1, IplImage * img2, vector<t_SURF_Pair> &pairs) {

    /*CvMat * mDist = cvCreateMat(1, pairs.size(), CV_32FC1);
    CvMat * mDistOrig = cvCreateMat(1, pairs.size(), CV_32FC1);

    CvPoint center1;
    CvPoint center2;
    for (int i = 0; i < pairs.size(); i++) {
        center1.x += pairs.at(i).kp1.pt.x;
        center1.y += pairs.at(i).kp1.pt.y;
        center2.x += pairs.at(i).kp2.pt.x;
        center2.y += pairs.at(i).kp2.pt.y;
    }
    center1.x /= pairs.size();
    center1.y /= pairs.size();
    center2.x /= pairs.size();
    center2.y /= pairs.size();

    for (int i = 0; i < pairs.size(); i++) {
        float dist1 = sqrt(pow(pairs.at(i).kp1.pt.x - center1.x, 2.0) + pow(pairs.at(i).kp1.pt.y - center1.y, 2.0));
        float dist2 = sqrt(pow(pairs.at(i).kp2.pt.x - center2.x, 2.0) + pow(pairs.at(i).kp2.pt.y - center2.y, 2.0));
        //float dist1 = pairs.at(i).kp1.pt.y - center1.y;
        //float dist2 = pairs.at(i).kp2.pt.y - center2.y;

        cvSetReal2D(mDist, 0, i, abs(dist1 - dist2));

        //double dist = min(dist1, dist2) / max(dist1, dist2);
        //cvSetReal2D(mDist, 0, i, dist);
    }


    cvNormalize(mDist, mDist, 0, 640, CV_MINMAX);
    cvCopy(mDist, mDistOrig);

    IplImage * imgDist = cvCreateImage(cvSize(pairs.size() * 2, 640), IPL_DEPTH_8U, 3);
    cvZero(imgDist);
    IplImage * img1C = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);
    IplImage * img2C = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 3);
    cvCvtColor(img1, img1C, CV_GRAY2BGR);
    cvCvtColor(img2, img2C, CV_GRAY2BGR);


    for (int i = 1; i < pairs.size(); i++) {
        if (i > 0)
            cvLine(imgDist, cvPoint(2 * (i-1), cvGetReal2D(mDist, 0, i-1)), cvPoint(2 * i, cvGetReal2D(mDist, 0, i)), cvScalar(0, 255, 0), 1, CV_AA);

        CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);

        cvCircle(img1C, pairs.at(i).kp1.pt, 2, color, -1, CV_AA);
        cvCircle(img2C, pairs.at(i).kp2.pt, 2, color, -1, CV_AA);
        cvCircle(imgDist, cvPoint(2 * i, cvGetReal2D(mDist, 0, i)), 2, color, -1, CV_AA);
    }

    // Ordenamos los valores
    vector<float> data;
    double minVal, maxVal;
    CvPoint minLoc;
    for (int i = 0; i < pairs.size(); i++) {
        cvMinMaxLoc(mDist, &minVal, &maxVal, &minLoc);

        data.push_back(minVal);

        cvSetReal2D(mDist, minLoc.y, minLoc.x, DBL_MAX);
    }          
    
    double median = 0;
    double q1 = 0;
    double q2 = 0;
    if (data.size() % 2 == 0) {
        int index = (data.size() / 2) - 1;
        median = (data.at(index) + data.at(index + 1)) / 2;
        index = (data.size() / 4) - 1;
        q1 = (data.at(index) + data.at(index + 1)) / 2;
        index = (data.size() * 3 / 4) - 1;
        q2 = (data.at(index) + data.at(index + 1)) / 2;
    } else {
        int index = data.size() / 2;
        median = data.at(index);
        index = data.size() / 4;
        q1 = data.at(index);
        index = data.size() * 3 / 4;
        q2 = data.at(index);
    }

    double k = 3.0;
    double maxThresh = q2 + k * (q2 - q1);
    double minThresh = q2 - k * (q2 - q1);        

    cvLine(imgDist, cvPoint(0, minThresh), cvPoint(2 * pairs.size(), minThresh), cvScalar(0, 0, 255), 1, CV_AA);
    cvLine(imgDist, cvPoint(0, maxThresh), cvPoint(2 * pairs.size(), maxThresh), cvScalar(0, 0, 255), 1, CV_AA);

    for (int i = 1; i < pairs.size(); i++) {
        //if ((cvGetReal2D(mDistOrig, 0, i) < minThresh) || (cvGetReal2D(mDistOrig, 0, i) > maxThresh))
        if (cvGetReal2D(mDistOrig, 0, i) > maxThresh) {
            cvCircle(img1C, pairs.at(i).kp1.pt, 3, cvScalar(0, 0, 255), 1, CV_AA);
            cvCircle(img2C, pairs.at(i).kp2.pt, 3, cvScalar(0, 0, 255), 1, CV_AA);
        } else {
            cvCircle(img1C, pairs.at(i).kp1.pt, 3, cvScalar(0, 255, 0), 1, CV_AA);
            cvCircle(img2C, pairs.at(i).kp2.pt, 3, cvScalar(0, 255, 0), 1, CV_AA);
        }
    }

    cvCircle(img1C, center1, 5, cvScalar(255, 0, 255), -1, CV_AA);
    cvCircle(img2C, center2, 5, cvScalar(255, 0, 255), -1, CV_AA);

    cvShowImage("dist", imgDist);
    cvShowImage("distImg1", img1C);
    cvShowImage("distImg2", img2C);

    vector<t_SURF_Pair> tmpPairs;
    for (int i = 0; i < pairs.size(); i++) {
        if (cvGetReal2D(mDistOrig, 0, i) < maxThresh)
            tmpPairs.push_back(pairs.at(i));
    }
    pairs = tmpPairs;

    cvReleaseImage(&imgDist);
    cvReleaseImage(&img1C);
    cvReleaseImage(&img2C);
    cvReleaseMat(&mDist);
    cvReleaseMat(&mDistOrig);

    /*int nPairs = pairs.size();
    CvMat * p1 = cvCreateMat(1, nPairs, CV_32FC2);
    CvMat * p2 = cvCreateMat(1, nPairs, CV_32FC2);
    CvMat * F = cvCreateMat(3, 3, CV_32FC1);
    CvMat *statusM = cvCreateMat(1, nPairs, CV_8UC1);

    for (int i = 0; i < nPairs; i++) {
        cvSet2D(p1, 0, i, cvScalar(pairs.at(i).kp1.pt.x, pairs.at(i).kp1.pt.y));
        cvSet2D(p2, 0, i, cvScalar(pairs.at(i).kp2.pt.x, pairs.at(i).kp2.pt.y));
    }

    cvFindFundamentalMat(p1, p2, F, method, 3., 0.70, statusM);*/
    //cvFindFundamentalMat(p1, p2, F, method, 3., 0.99, statusM);

    //removeOutliers(&p1, &p2, statusM);

    /*int size = pairs.size() * 2;

    double * a = new double[size * 8];
    double * b = new double[size];
    double * x = new double[9];

    CvMat A, B, F;
    A = cvMat(size, 8, CV_64FC1, a);
    B = cvMat(size, 1, CV_64FC1, b);
    F = cvMat(8, 1, CV_64FC1, x);

    for (int i = 0; i < pairs.size(); i++) {
        a[i * 8] = a[(i + pairs.size()) * 8 + 3] = pairs.at(i).kp1.pt.x;
        a[i * 8 + 1] = a[(i + pairs.size()) * 8 + 4] = pairs.at(i).kp1.pt.y;
        a[i * 8 + 2] = a[(i + pairs.size()) * 8 + 5] = 1;
        a[i * 8 + 3] = a[i * 8 + 4] = a[i * 8 + 5] =
                a[(i + pairs.size()) * 8] = a[(i + pairs.size()) * 8 + 1] = a[(i + pairs.size()) * 8 + 2] = 0;
        a[i * 8 + 6] = -pairs.at(i).kp1.pt.x * pairs.at(i).kp2.pt.x;
        a[i * 8 + 7] = -pairs.at(i).kp1.pt.y * pairs.at(i).kp2.pt.x;
        a[(i + pairs.size()) * 8 + 6] = -pairs.at(i).kp1.pt.x * pairs.at(i).kp2.pt.y;
        a[(i + pairs.size()) * 8 + 7] = -pairs.at(i).kp1.pt.y * pairs.at(i).kp2.pt.y;
        b[i] = pairs.at(i).kp2.pt.x;
        b[i + pairs.size()] = pairs.at(i).kp2.pt.y;
    }

    cvSolve(&A, &B, &F, CV_SVD);

    x[8] = 1;

    F = cvMat(3, 3, CV_64FC1, x);

    IplImage * transf = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);
    cvZero(transf);

    for (int i = 0; i < pairs.size(); i++) {
        CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);

        cvCircle(transf, pairs.at(i).kp1.pt, 2, color, -1, CV_AA);
        cvCircle(transf, pairs.at(i).kp2.pt, 2, color, -1, CV_AA);

        cvCircle(transf, pairs.at(i).kp1.pt, 3, cvScalar(0, 0, 255), 1, CV_AA);
        cvCircle(transf, pairs.at(i).kp2.pt, 3, cvScalar(0, 255, 0), 1, CV_AA);
    }

    cvShowImage("transf1", transf);

    CvMat * dest = cvCreateMat(pairs.size(), 1, CV_32FC2);
    CvMat * orig = cvCreateMat(pairs.size(), 1, CV_32FC2);
    for (int i = 0; i < pairs.size(); i++) {
        cvSet2D(orig, i, 0, cvScalar(pairs.at(i).kp1.pt.y, pairs.at(i).kp1.pt.x));
    }

    cvPerspectiveTransform(orig, dest, &F);

    for (int i = 0; i < pairs.size(); i++) {
        CvScalar tmp = cvGet2D(dest, i, 0);
        pairs.at(i).kp1.pt.x = tmp.val[0];
        pairs.at(i).kp1.pt.y = tmp.val[1];

        //cout << tmp.val[0] << ", " << tmp.val[1] << endl;
    }

    cvReleaseMat(&orig);
    cvReleaseMat(&dest);

    cvZero(transf);

    for (int i = 0; i < pairs.size(); i++) {
        CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);

        cvCircle(transf, pairs.at(i).kp1.pt, 2, color, -1, CV_AA);
        //cvCircle(transf, pairs.at(i).kp2.pt, 2, color, -1, CV_AA);

        //cvCircle(transf, pairs.at(i).kp1.pt, 2, cvScalar(0, 0, 255), 1, CV_AA);
        //cvCircle(transf, pairs.at(i).kp2.pt, 2, cvScalar(0, 255, 0), 1, CV_AA);
    }

    cvShowImage("transf", transf);

    cvWaitKey(0);

    cvReleaseImage(&transf);

    /*nPairs = p1->cols;
    pairs.clear();
    CvScalar pA, pB;
    for (int i = 0; i < nPairs; i++) {
        pA = cvGet2D(p1, 0, i);
        pB = cvGet2D(p2, 0, i);

        t_SURF_Pair pair;
        pair.kp1.pt = cvPoint2D32f(pA.val[0], pA.val[1]);
        pair.kp2.pt = cvPoint2D32f(pB.val[0], pB.val[1]);
        pairs.push_back(pair);
    }*/

    //cvReleaseMat(&p1);
    //cvReleaseMat(&p2);
    //cvReleaseMat(&F);
    //cvReleaseMat(&statusM);

    vector <int> nearest1;
    vector <int> nearest2;

    for (int i = 0; i < pairs.size(); i++) {
        float minDist1 = DBL_MAX;
        float minDist2 = DBL_MAX;
        int minIdx1 = -1;
        int minIdx2 = -1;
        for (int j = 0; j < pairs.size(); j++) {
            if (i == j)
                continue;

            float dist1 = sqrt(pow(pairs.at(i).kp1.pt.x - pairs.at(j).kp1.pt.x, 2.0) + pow(pairs.at(i).kp1.pt.y - pairs.at(j).kp1.pt.y, 2.0));
            float dist2 = sqrt(pow(pairs.at(i).kp2.pt.x - pairs.at(j).kp2.pt.x, 2.0) + pow(pairs.at(i).kp2.pt.y - pairs.at(j).kp2.pt.y, 2.0));

            if (dist1 < minDist1) {
                minIdx1 = j;
                minDist1 = dist1;
            }
            if (dist2 < minDist2) {
                minIdx2 = j;
                minDist2 = dist2;
            }
        }

        nearest1.push_back(minIdx1);
        nearest2.push_back(minIdx2);
    }

    IplImage * img1C = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);
    IplImage * img2C = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 3);
    cvCvtColor(img1, img1C, CV_GRAY2BGR);
    cvCvtColor(img2, img2C, CV_GRAY2BGR);

    for (int i = 1; i < pairs.size(); i++) {
        CvScalar color = cvScalar(rand() & 255, rand() & 255, rand() & 255);

        cvCircle(img1C, pairs.at(i).kp1.pt, 2, color, -1, CV_AA);
        cvCircle(img2C, pairs.at(i).kp2.pt, 2, color, -1, CV_AA);

        if (nearest1.at(i) != nearest2.at(i)) {
            cvCircle(img1C, pairs.at(i).kp1.pt, 3, cvScalar(0, 0, 255), 1, CV_AA);
            cvCircle(img2C, pairs.at(i).kp2.pt, 3, cvScalar(0, 0, 255), 1, CV_AA);
        } else {
            cvCircle(img1C, pairs.at(i).kp1.pt, 3, cvScalar(0, 255, 0), 1, CV_AA);
            cvCircle(img2C, pairs.at(i).kp2.pt, 3, cvScalar(0, 255, 0), 1, CV_AA);
        }
    }

    cvShowImage("distImg1", img1C);
    cvShowImage("distImg2", img2C);

    vector<t_SURF_Pair> tmpPairs;
    for (int i = 0; i < pairs.size(); i++) {
        if (nearest1.at(i) == nearest2.at(i))
            tmpPairs.push_back(pairs.at(i));
    }
    pairs = tmpPairs;


    cvReleaseImage(&img1C);
    cvReleaseImage(&img2C);
}

inline void SurfGPU::cleanRANSAC(int method, vector<t_SURF_Pair> &pairs) {
    int nPairs = pairs.size();
    CvMat * p1 = cvCreateMat(1, nPairs, CV_32FC2);
    CvMat * p2 = cvCreateMat(1, nPairs, CV_32FC2);
    CvMat * F = cvCreateMat(3, 3, CV_32FC1);
    CvMat *statusM = cvCreateMat(1, nPairs, CV_8UC1);

    for (int i = 0; i < nPairs; i++) {
        cvSet2D(p1, 0, i, cvScalar(pairs.at(i).kp1.pt.x, pairs.at(i).kp1.pt.y));
        cvSet2D(p2, 0, i, cvScalar(pairs.at(i).kp2.pt.x, pairs.at(i).kp2.pt.y));
    }

    cvFindFundamentalMat(p1, p2, F, method, 3., 0.70, statusM);
    //cvFindFundamentalMat(p1, p2, F, method, 3., 0.99, statusM);

    removeOutliers(&p1, &p2, statusM);

    nPairs = p1->cols;
    pairs.clear();
    CvScalar pA, pB;
    for (int i = 0; i < nPairs; i++) {
        pA = cvGet2D(p1, 0, i);
        pB = cvGet2D(p2, 0, i);

        t_SURF_Pair pair;
        pair.kp1.pt = cvPoint2D32f(pA.val[0], pA.val[1]);
        pair.kp2.pt = cvPoint2D32f(pB.val[0], pB.val[1]);
        pairs.push_back(pair);
    }

    cvReleaseMat(&p1);
    cvReleaseMat(&p2);
    cvReleaseMat(&F);
    cvReleaseMat(&statusM);
}

inline double SurfGPU::getCorrelation(vector <float> data1, vector<float> data2) {

    double mean1 = 0.0;
    double mean2 = 0.0;

    for (int i = 0; i < data1.size(); i++) {
        mean1 += data1.at(i);
        mean2 += data2.at(i);
    }
    mean1 /= data1.size();
    mean2 /= data2.size();

    double sdv1 = 0.0;
    double sdv2 = 0.0;

    for (int i = 0; i < data1.size(); i++) {
        sdv1 += data1.at(i) - mean1;
        sdv2 += data2.at(i) - mean2;
    }
    sdv1 /= data1.size() - 1;
    sdv2 /= data2.size() - 1;

    sdv1 = sqrt(sdv1);
    sdv2 = sqrt(sdv2);

    double corr = 0.0;
    for (int i = 0; i < data1.size(); i++) {
        corr += (data1.at(i) - mean1) * (data2.at(i) - mean2);
    }
    corr /= sdv1 * sdv2 * (data1.size() - 1);
}

inline void SurfGPU::cleanByCorrelation(vector<t_SURF_Pair> &pairs, IplImage * img1, IplImage * img2) {
    vector<t_SURF_Pair> tmpPairs;
    for (vector<t_SURF_Pair>::iterator it = pairs.begin(); it != pairs.end(); it++) {
        float corr = calcCCorr(img1, img2, *it, false);
        if (abs(corr) > 0.5)
            tmpPairs.push_back(*it);
    }
    pairs = tmpPairs;
}

inline double SurfGPU::calcCCorr(IplImage * img1, IplImage * img2, t_SURF_Pair pair, bool show) {
    //CvPoint2D32f pts1[4], pts2[4], destPoints[4];
    CvPoint2D32f destPoints[4];

    CvPoint2D32f * pts1 = getSquare(pair.kp1);
    CvPoint2D32f * pts2 = getSquare(pair.kp2);

    CvSize newSize = cvSize(max(2 * pair.kp1.size, 2 * pair.kp2.size), max(2 * pair.kp1.size, 2 * pair.kp2.size));
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
    IplImage * mask = cvCreateImage(newSize, IPL_DEPTH_8U, 1);

    cvZero(mask);
    CvPoint center = cvPoint(max(pair.kp1.size, pair.kp2.size), max(pair.kp1.size, pair.kp2.size));
    cvCircle(mask, center, (int)max(pair.kp1.size, pair.kp2.size), cvScalar(255), -1, CV_AA);

    cvWarpPerspective(img1, dest1, mat1, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
    cvWarpPerspective(img2, dest2, mat2, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, cvScalarAll(0));

    if (show) {
        cvNamedWindow("Dest1", 1);
        cvNamedWindow("Dest2", 1);
        cvNamedWindow("DestMask", 1);

        cvShowImage("Dest1", dest1);
        cvShowImage("Dest2", dest2);
        cvShowImage("DestMask", mask);

        //cvWaitKey(0);
    }

    CvScalar mean1, std1, mean2, std2;
    cvAvgSdv(dest1, &mean1, &std1);
    cvAvgSdv(dest2, &mean2, &std2);

    double corr = 0;
    for (int i = 0; i < newSize.height; i++) {
        for (int j = 0; j < newSize.width; j++) {
            if (cvGetReal2D(mask, i, j) == 255)
                corr += (cvGetReal2D(dest1, i, j) - mean1.val[0]) * (cvGetReal2D(dest2, i, j) - mean2.val[0]);
        }
    }
    corr /= (cvCountNonZero(mask) - 1) * std1.val[0] * std2.val[0];

//    cvWaitKey(0);

    cvReleaseImage(&dest1);
    cvReleaseImage(&dest2);
    cvReleaseImage(&mask);

    cvReleaseMat(&mat1);
    cvReleaseMat(&mat2);

    return corr;
}


inline float SurfGPU::distSquare(vector <float> v1, vector <float> v2) {
	double dsq = 0.;

        for (vector<float>::iterator it1 = v1.begin(); it1 != v1.end(); it1++) {
            for (vector<float>::iterator it2 = v2.begin(); it2 != v1.end(); it2++) {
                dsq += (*it1 - *it2) * (*it1 - *it2);
            }
        }
	return dsq;
}

// Find closest interest point in a list, given one interest point
/*int CRealMatches::SurfGPU(const KeyPoint& kp1, const vector< ISurfPoint >& ipts, int vlen) {
	double mind = 1e100, second = 1e100;
	int match = -1;

	for (unsigned i = 0; i < ipts.size(); i++) {

		// Take advantage of Laplacian to speed up matching
		if (ipts[i].laplace != ip1.laplace)
			continue;

		double d = distSquare(ipts[i].ivec, ip1.ivec, vlen);

		if (d < mind) {
			second = mind;
			mind = d;
			match = i;
		} else if (d < second) {
			second = d;
		}

	}

	if (mind < 0.8 * second)
		return match;

	return -1;
}

// Find all possible matches between two images
vector< int > SurfGPU::findMatches(const vector< ISurfPoint >& ipts1, const vector< ISurfPoint >& ipts2, int vlen) {
	vector< int > matches(ipts1.size());
	int c = 0;
	for (unsigned i = 0; i < ipts1.size(); i++) {
		int match = findMatch(ipts1[i], ipts2, vlen);
		matches[i] = match;
		if (match != -1) {
                    c++;
		}
	}

        return matches;
}//*/

void SurfGPU::matchSequential(vector<KeyPoint> points1, vector<KeyPoint> points2,  vector<float> desc1, vector<float> desc2, vector <t_SURF_Pair> &pairs) {

    for (int i = 0; i < points1.size(); i++) {
        KeyPoint kp1 = points1.at(i);
        //vector<float> descriptor1(desc1.begin() + i * 64, desc1.begin() + ((i + 1) * 64));
        vector<float> descriptor1;
        vector<float>::const_iterator it1 = desc1.begin() + (i * 64);
        for(int k = 0; k < 64; k++, it1++) {
            descriptor1.push_back(*it1);
        }

        double mind = DBL_MAX, second = DBL_MAX;
        int match = -1;
        
        for (int j = 0; j < points2.size(); j++) {
            KeyPoint kp2 = points2.at(j);
            //vector<float> descriptor2(desc2.begin() + j * 64, desc2.begin() + ((j + 1) * 64));
            vector<float> descriptor2;
            vector<float>::const_iterator it2 = desc2.begin() + (j * 64);
            for(int k = 0; k < 64; k++, it2++) {
                descriptor2.push_back(*it2);
            }

            //if (isLastBitSet(kp1.response) != isLastBitSet(kp2.response))
            //    continue;

            double d = distSquare(descriptor1, descriptor2);

            if (d < mind) {
                second = mind;
		mind = d;
		match = j;
            } else if (d < second) {
                second = d;
            }
        }

        if (mind < 0.8 * second) {
            t_SURF_Pair pair;
            pair.kp1 = kp1;
            pair.kp2 = points2.at(match);
            //pair.desc1 = descriptor1;
            //vector<float> descriptor2(desc2.begin() + match * 64, desc2.begin() + ((match + 1) * 64));
            //pair.desc2 = descriptor2;

            pairs.push_back(pair);
            
        }        
        //cout << "Salio" << endl;
    }
    cout << "Pairs = " << pairs.size() << endl;
}//*/

inline void SurfGPU::setMaskFromPoints(IplImage * &mask, vector<t_SURF_Pair> pairs, int index) {
    CvPoint* pts = (CvPoint*) malloc(pairs.size() * sizeof (CvPoint));
    int* hull = (int*) malloc(pairs.size() * sizeof (hull[0]));
    CvMat point_mat = cvMat(1, pairs.size(), CV_32SC2, pts);
    CvMat hull_mat = cvMat(1, pairs.size(), CV_32SC1, hull);

    CvPoint pt;
    if (index == 0) {
        for (int i = 0; i < pairs.size(); i++) {
            pts[i] = cvPointFrom32f(pairs.at(i).kp1.pt);
        }
    } else {
        for (int i = 0; i < pairs.size(); i++) {
            pts[i] = cvPointFrom32f(pairs.at(i).kp2.pt);
        }
    }

    cvConvexHull2(&point_mat, &hull_mat, CV_CLOCKWISE, 0);
    int hullcount = hull_mat.cols;

    pt = pts[hull[hullcount - 1]];

    CvPoint * poly = new CvPoint[hullcount];
    for (int i = 0; i < hullcount; i++) {
        poly[i] = pt;
        pt = pts[hull[i]];
    }

    cvZero(mask);
    cvFillConvexPoly(mask, poly, hullcount, cvScalar(255));

    //cvErode(mask, mask);

    delete pts;
    delete hull;
    delete poly;
}

inline void SurfGPU::cleanByPosition(vector<t_SURF_Pair> pairs, CvSize size) {
    IplImage * mask1 = cvCreateImage(size, IPL_DEPTH_8U, 1);
    IplImage * mask2 = cvCreateImage(size, IPL_DEPTH_8U, 1);

    setMaskFromPoints(mask1, pairs, 0);
    setMaskFromPoints(mask2, pairs, 1);

    drawPairs(pairs, mask1, mask2);

    cvReleaseImage(&mask1);
    cvReleaseImage(&mask2);
}

void SurfGPU::testSurf(string file1, string file2) {
    //config.threshold = 0.01f;

    IplImage * img1In = cvLoadImage(file1.c_str(), 0);
    IplImage * img2In = cvLoadImage(file2.c_str(), 0);

    IplImage * img1 = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 1);
    IplImage * img2 = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 1);

    cvResize(img1In, img1);
    cvResize(img2In, img2);

    cvReleaseImage(&img1In);
    cvReleaseImage(&img2In);

    vector<KeyPoint> points1;
    vector<KeyPoint> points2;

    vector<float> desc1;
    vector<float> desc2;

    vector <t_SURF_Pair> pairs;

    //for (int i = 0; i < 3; i++) {
        clock_t myTime = clock();
        testSurfGPU(img1, points1, desc1, config);
        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo img1 = " << time << endl;
        myTime = clock();
        testSurfGPU(img2, points2, desc2, config);
        time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo img2 = " << time << endl;
    //}

    //matchSequential(points2, points1, desc2, desc1, pairs);
    //for (int i = 0; i < 3; i++) {
        //clock_t myTime = clock();
        bruteMatch(points1, points2, desc1, desc2, pairs);
        cleanRANSAC(CV_FM_RANSAC, pairs);
        //time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        //cout << "Tiempo match = " << time << endl;
    //}
    //if (pairs.size() == 0)
    //    exit(0);

    //cleanByPosition(pairs, cvSize(640, 480));
    
    //drawKeypoints(points1, img1, "surfImg1");
    //drawKeypoints(points2, img2, "surfImg2");

    cleanDistances(img1, img2, pairs);
    //cleanByCorrelation(pairs, img1, img2);

    //drawPairs(pairs, img1, img2);//*/

    /*t_moment * moments1, * moments2;
    int nMoments1, nMoments2;
    myTime = clock();
    mesrTest(img1, "mser1", moments1, nMoments1);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo img1 = " << time << endl;
    myTime = clock();
    mesrTest(img2, "mser2", moments2, nMoments2);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo img2 = " << time << endl;

    
    vector<t_moment *> regionPairs;
    matchMserByMoments(img1, img2, moments1, moments2, nMoments1, nMoments2, "Match", regionPairs);
    cout << "Detectados " << regionPairs.size() << endl;//*/

    int key = cvWaitKey(0);

    cvReleaseImage(&img1);
    cvReleaseImage(&img2);
}

void SurfGPU::testSurf(IplImage * img1, IplImage * img2, vector <t_SURF_Pair> &pairs) {
    vector<KeyPoint> points1;
    vector<KeyPoint> points2;

    vector<float> desc1;
    vector<float> desc2;

    clock_t myTime = clock();    
    testSurfGPU(img1, points1, desc1, config);
    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo img1 = " << time << endl;
    myTime = clock();
    testSurfGPU(img2, points2, desc2, config);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo img2 = " << time << endl;   
    
    myTime = clock();
    if ((points1.size() == 0) || (points2.size() == 0)) {
        pairs.clear();
    } else {
        bruteMatch(points1, points2, desc1, desc2, pairs);
        //cleanByCorrelation(pairs, img1, img2);

        if (pairs.size() > 8) {
            cleanRANSAC(CV_FM_RANSAC, pairs);
            //cleanRANSAC(CV_FM_RANSAC, pairs);
        }
        //cleanDistances(img1, img2, pairs);
        time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo match = " << time << endl;//*/
    }

    //drawPairs(pairs, img1, img2);

    //cleanByPosition(pairs, cvGetSize(img1));
}