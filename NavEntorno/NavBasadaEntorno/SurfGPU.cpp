/* 
 * File:   SurfGPU.cpp
 * Author: nestor
 * 
 * Created on 25 de junio de 2010, 12:39
 */

#include "SurfGPU.h"
#include "CRealMatches.h"

SurfGPU::SurfGPU() {
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
        cvCircle(imgColor1, pair->kp1.pt, 2, color, -1, CV_AA);

        Point2f dir2(pair->kp2.pt);
        st = pair->kp2.size * sin(pair->kp2.angle);
        ct = pair->kp2.size * cos(pair->kp2.angle);
        dir2.x += ct;
        dir2.y += st;
        //cvCircle(imgColor2, pair->kp2.pt, (int) pair->kp2.size, color, 1, CV_AA);
        //cvLine(imgColor2, pair->kp2.pt, dir2, color, 1, CV_AA);
        cvCircle(imgColor2, pair->kp2.pt, 2, color, -1, CV_AA);
    }

    cvShowImage("surfMatch1", imgColor1);
    cvShowImage("surfMatch2", imgColor2);

    cvReleaseImage(&imgColor1);
    cvReleaseImage(&imgColor2);
}

void SurfGPU::bruteMatch(vector<KeyPoint> points1, vector<KeyPoint> points2, vector<float> desc1, vector<float> desc2, vector<t_SURF_Pair> &pairs) {
    double* avg1 = (double*) malloc(sizeof (double) * points1.size());
    double* avg2 = (double*) malloc(sizeof (double) * points2.size());
    double* dev1 = (double*) malloc(sizeof (double) * points1.size());
    double* dev2 = (double*) malloc(sizeof (double) * points2.size());

    int* best1 = (int*) malloc(sizeof (int) * points1.size());
    int* best2 = (int*) malloc(sizeof (int) * points2.size());

    double* best1corr = (double*) malloc(sizeof (double) * points1.size());
    double* best2corr = (double*) malloc(sizeof (double) * points2.size());
    
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
    double corr;
    for (int i = 0; i < points1.size(); i++) {
        vector<float> descriptor1;
        for (int k = i * descriptor_size; k < (i * descriptor_size) + descriptor_size; k++)
            descriptor1.push_back(desc1.at(k));
        // seq1 = (float*) cvGetSeqElem(desc1, i); //seq1 es el descriptor1
        for (int j = 0; j < points2.size(); j++) {
            vector<float> descriptor2;
            for (int k = j * descriptor_size; k < (j * descriptor_size) + descriptor_size; k++)
                descriptor2.push_back(desc2.at(k));

            corr = 0;
            // seq2 = (float*) cvGetSeqElem(desc1, i); //seq2 es el descriptor2
            for (int k = 0; k < descriptor_size; k++)
                corr += (descriptor1.at(k) - avg1[i])*(descriptor2.at(k) - avg2[j]);
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
    cout << endl;

    double CORRELATION_THRESHOLD = 0.75;
    for (int i = 0; i < points1.size(); i++) {
        if (best2[best1[i]] == i && best1corr[i] > CORRELATION_THRESHOLD) {
            t_SURF_Pair pair;
            pair.kp1 = points1.at(i);
            pair.kp2 = points2.at(best1[i]);
            pairs.push_back(pair);
        }
    }//*/
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

    for (int i = 0; i < 3; i++) {
        clock_t myTime = clock();
        testSurfGPU(img1, points1, desc1, config);
        time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo img1 = " << time << endl;
        myTime = clock();
        testSurfGPU(img2, points2, desc2, config);
        time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
        cout << "Tiempo img2 = " << time << endl;
    }

    //matchSequential(points2, points1, desc2, desc1, pairs);
    clock_t myTime = clock();
    bruteMatch(points1, points2, desc1, desc2, pairs);
    cleanRANSAC(CV_FM_RANSAC, pairs);
    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo match = " << time << endl;
    //if (pairs.size() == 0)
    //    exit(0);
    
    drawKeypoints(points1, img1, "surfImg1");
    drawKeypoints(points2, img2, "surfImg2");
    drawPairs(pairs, img1, img2);//*/

    t_moment * moments1, * moments2;
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
    bruteMatch(points1, points2, desc1, desc2, pairs);
    //cleanRANSAC(CV_FM_RANSAC, pairs);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo match = " << time << endl;

    //drawPairs(pairs, img1, img2);
}