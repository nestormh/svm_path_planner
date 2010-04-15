#include "CRealMatches.h"

#define THRESH 100+100

void CRealMatches::wm(vector<t_Pair> pairs, IplImage * img1, IplImage * img2) {
    int nMax = 10;

    CvMat * remapX = cvCreateMat(size.height, size.width, CV_32FC1);
    CvMat * remapY = cvCreateMat(size.height, size.width, CV_32FC1);
    IplImage * wm = cvCreateImage(size, IPL_DEPTH_8U, 1);

    map<int, vector<t_DistDesc> > distances;
    clock_t myTime = clock();
    getDistances(pairs, distances, nMax);
    time_t time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en getDistances = " << time << endl;
    /*for (map<int, vector<t_DistDesc> >::iterator it = distances.begin(); it != distances.end(); it++) {
        cout << pairs.at(it->first).p1.x << ", " << pairs.at(it->first).p1.y << " >> ";
        for (vector<t_DistDesc>::iterator it2 = it->second.begin(); it2 != it->second.end(); it2++) {
            cout << it2->p.x << ", " << it2->p.y << " :: ";
        }
        cout << endl;
    }//*/    

    double * coefs1, * coefs2;
    myTime = clock();
    //calculateCoefsMQ(pairs, coefs1, coefs2);
    calculateCoefsWM(pairs, coefs1, coefs2, distances, nMax);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en calculateCoefs = " << time << endl;

    /*myTime = clock();
    vector <double *> polynomialsX;
    vector <double *> polynomialsY;
    getPolynomials(pairs, polynomialsX, polynomialsY, distances, coefs1, coefs2);
    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en getPolynomials = " << time << endl;//*/

    double u, v;
    myTime = clock();
    for (int i = 0; i < size.height; i++) {
        for (int j = 0; j < size.width; j++) {
            //getValueByCoefs(cvPoint2D32f(j, i), u, v, pairs, distances, polynomialsX, polynomialsY);
            getValueByCoefsWM(cvPoint2D32f(j, i), u, v, pairs, distances, coefs1, coefs2, nMax);
            //getValueByCoefsMQ(cvPoint2D32f(j, i), u, v, pairs, coefs1, coefs2);            

            cvSetReal2D(remapX, i, j, u);
            cvSetReal2D(remapY, i, j, v);
        }
    }

    time = (double(clock() - myTime) / CLOCKS_PER_SEC * 1000);
    cout << "Tiempo invertido en getValueByCoefs = " << time << endl;

    cvRemap(img2, wm, remapX, remapY, CV_INTER_AREA+CV_WARP_FILL_OUTLIERS, cvScalar(0));
    cvNamedWindow("WM", 1);
    cvShowImage("WM", wm);

    cvReleaseMat(&remapX);
    cvReleaseMat(&remapY);
    cvReleaseImage(&wm);
    delete coefs1;
    delete coefs2;
}

inline void CRealMatches::getDistancesVector(int index, vector<t_Pair> pairs, vector<t_DistDesc> &points, int nMax) {
    for (int i = 0; i < pairs.size(); i++) {
        if (i == index) continue;

        double a = pairs.at(index).p1.x - pairs.at(i).p1.x;
        double b = pairs.at(index).p1.y - pairs.at(i).p1.y;
        double dist = a * a + b * b;
        if (dist == 0) continue;

        vector<t_DistDesc>::iterator pos = points.begin();
        for (; pos != points.end(); pos++) {
            if (pos->dist > dist)
                break;
        }
        t_DistDesc desc;
        desc.dist = sqrt(dist);
        desc.p = pairs.at(i).p1;
        desc.index = i;
        points.insert(pos, desc);
    }
    while (points.size() > nMax) {
        points.pop_back();
    }
}

inline void CRealMatches::getDistancesVector(CvPoint2D32f p, vector<t_Pair> pairs, vector<t_DistDesc> &points, int nMax) {
    for (int i = 0; i < pairs.size(); i++) {
        double a = p.x - pairs.at(i).p1.x;
        double b = p.y - pairs.at(i).p1.y;
        double dist = a * a + b * b;
        if (dist == 0) continue;

        vector<t_DistDesc>::iterator pos = points.begin();
        for (; pos != points.end(); pos++) {
            if (pos->dist > dist)
                break;
        }
        t_DistDesc desc;
        desc.dist = sqrt(dist);
        desc.p = pairs.at(i).p1;
        desc.index = i;
        points.insert(pos, desc);
    }
    while (points.size() > nMax) {
        points.pop_back();
    }
}

inline void CRealMatches::getDistances(vector<t_Pair> pairs, map<int, vector<t_DistDesc> > &distances, int nMax) {
    for (int i = 0; i < pairs.size(); i++) {
        vector<t_DistDesc> points;
        getDistancesVector(i, pairs, points, nMax);        
        distances.insert(make_pair(i, points));
    }
}

inline void CRealMatches::calculateCoefs(vector<t_Pair> pairs, double * &coefs1, double * &coefs2, map<int, vector<t_DistDesc> > distances) {
    CvMat * A = cvCreateMat(pairs.size(), pairs.size(), CV_64FC1);
    CvMat * Bu = cvCreateMat(pairs.size(), 1, CV_64FC1);
    coefs1 = new double[pairs.size()];
    CvMat Xu = cvMat(pairs.size(), 1, CV_64FC1, coefs1);
    CvMat * Bv = cvCreateMat(pairs.size(), 1, CV_64FC1);
    coefs2 = new double[pairs.size()];
    CvMat Xv = cvMat(pairs.size(), 1, CV_64FC1, coefs2);
    double d2 = 0;

    for (int i = 0; i < pairs.size(); i++) {
        double rn = distances.at(i).at(distances.at(i).size() -1).dist;
        double sumR = 0;
        for (int j = 0; j < pairs.size(); j++) {
            double a = pairs.at(i).p1.x - pairs.at(j).p1.x;
            double b = pairs.at(i).p1.y - pairs.at(j).p1.y;
            double r = sqrt(a * a + b * b + d2);
            if (r < rn) {
                cvSetReal2D(A, i, j, 0);
            } else {
                r /= rn;
                double R = 1 - 3 * r * r + 2 * r * r * r; // 1 - 3r² + 2r³
                sumR += R;
                cvSetReal2D(A, i, j, R);
            }
        }
        for (int j = 0; j < pairs.size(); j++) {
            //cvSetReal2D(A, i, j, cvGetReal2D(A, i, j) / sumR);
        }
        cvSetReal2D(Bu, i, 0, pairs.at(i).p2.x);
        cvSetReal2D(Bv, i, 0, pairs.at(i).p2.y);
    }
    cvSolve(A, Bu, &Xu, CV_SVD);
    cvSolve(A, Bv, &Xv, CV_SVD);

    cvReleaseMat(&A);
    cvReleaseMat(&Bu);
    cvReleaseMat(&Bv);
}

inline void CRealMatches::calculateCoefsWM(vector<t_Pair> pairs, double * &coefs1, double * &coefs2, map<int, vector<t_DistDesc> > distances, int nMax) {
    CvMat * A = cvCreateMat(pairs.size(), nMax, CV_64FC1);
    CvMat * Bu = cvCreateMat(pairs.size(), 1, CV_64FC1);
    coefs1 = new double[nMax];
    CvMat Xu = cvMat(nMax, 1, CV_64FC1, coefs1);
    CvMat * Bv = cvCreateMat(pairs.size(), 1, CV_64FC1);
    coefs2 = new double[nMax];
    CvMat Xv = cvMat(nMax, 1, CV_64FC1, coefs2);
    double d2 = 0;

    for (int i = 0; i < pairs.size(); i++) {
        //double rn = distances.at(i).at(distances.at(i).size() - 1).dist;
        double sumR = 0;
        for (int j = 0; j < nMax; j++) {
            double a = pairs.at(i).p1.x - distances.at(i).at(j).p.x;
            double b = pairs.at(i).p1.y - distances.at(i).at(j).p.y;
            double r = 1/sqrt(a * a + b * b + d2);
            sumR += r;
            cvSetReal2D(A, i, j, r);
        }
        for (int j = 0; j < nMax; j++) {
            cvSetReal2D(A, i, j, cvGetReal2D(A, i, j) / sumR);
        }//*/
        cvSetReal2D(Bu, i, 0, pairs.at(i).p2.x);
        cvSetReal2D(Bv, i, 0, pairs.at(i).p2.y);
    }
    cvSolve(A, Bu, &Xu, CV_SVD);
    cvSolve(A, Bv, &Xv, CV_SVD);

    cvReleaseMat(&A);
    cvReleaseMat(&Bu);
    cvReleaseMat(&Bv);
}

inline void CRealMatches::calculateCoefsMQ(vector<t_Pair> pairs, double * &coefs1, double * &coefs2) {
    CvMat * A = cvCreateMat(pairs.size(), pairs.size(), CV_64FC1);
    CvMat * Bu = cvCreateMat(pairs.size(), 1, CV_64FC1);
    coefs1 = new double[pairs.size()];
    CvMat Xu = cvMat(pairs.size(), 1, CV_64FC1, coefs1);
    CvMat * Bv = cvCreateMat(pairs.size(), 1, CV_64FC1);
    coefs2 = new double[pairs.size()];
    CvMat Xv = cvMat(pairs.size(), 1, CV_64FC1, coefs2);
    double d2 = 0;

    for (int i = 0; i < pairs.size(); i++) {
        for (int j = 0; j < pairs.size(); j++) {
            double a = pairs.at(i).p1.x - pairs.at(j).p1.x;
            double b = pairs.at(i).p1.y - pairs.at(j).p1.y;
            double r = a * a + b * b + d2;
            cvSetReal2D(A, i, j, sqrt(r));            
        }
        cvSetReal2D(Bu, i, 0, pairs.at(i).p2.x);
        cvSetReal2D(Bv, i, 0, pairs.at(i).p2.y);
    }
    cvSolve(A, Bu, &Xu, CV_SVD);
    cvSolve(A, Bv, &Xv, CV_SVD);

    cvReleaseMat(&A);
    cvReleaseMat(&Bu);
    cvReleaseMat(&Bv);
}

inline void CRealMatches::getPolynomials(vector<t_Pair> pairs, vector <double *> &polynomialsX, vector <double *> &polynomialsY, map<int, vector<t_DistDesc> > distances, double * coefs1, double * coefs2) {
    for (int i = 0; i < pairs.size(); i++) {
        int index1 = distances.at(i).at(0).index;
        int index2 = distances.at(i).at(1).index;
        double * polyX;
        double * polyY;
        CvPoint3D32f o = cvPoint3D32f(pairs.at(i).p1.x, pairs.at(i).p1.y, pairs.at(i).p2.x);        
        CvPoint3D32f p1 = cvPoint3D32f(pairs.at(index1).p1.x, pairs.at(index1).p1.y, pairs.at(index1).p2.x);
        CvPoint3D32f p2 = cvPoint3D32f(pairs.at(index2).p1.x, pairs.at(index2).p1.y, pairs.at(index2).p2.x);
        polyX = getPolynoms(o, p1, p2, coefs1[i]);
        polynomialsX.push_back(polyX);
        o = cvPoint3D32f(pairs.at(i).p1.x, pairs.at(i).p1.y, pairs.at(i).p2.y);
        p1 = cvPoint3D32f(pairs.at(index1).p1.x, pairs.at(index1).p1.y, pairs.at(index1).p2.y);
        p2 = cvPoint3D32f(pairs.at(index2).p1.x, pairs.at(index2).p1.y, pairs.at(index2).p2.y);
        polyY = getPolynoms(o, p1, p2, coefs2[i]);
        polynomialsY.push_back(polyY);
    }
}

inline double * CRealMatches::getPolynoms(CvPoint3D32f o, CvPoint3D32f p1, CvPoint3D32f p2, double coef) {
    double poly[3];

    CvPoint3D32f u = cvPoint3D32f(p1.x - o.x, p1.y - o.y, p1.z - o.z);
    CvPoint3D32f v = cvPoint3D32f(p2.x - o.x, p2.y - o.y, p2.z - o.z);

    double detA = u.y * v.z - v.y * u.z;
    double detB = -(u.x * v.z - v.x * u.z);
    double detC = u.x * v.x - v.x * u.y;

    double D = -detA * o.x - detB * o.y - detC * o.z - coef;

    poly[0] = detA / -detC;
    poly[1] = detB / -detC;
    poly[2] = D / -detC;

    return poly;
}

inline void CRealMatches::getValueByCoefs(CvPoint2D32f p, double &u, double &v, vector<t_Pair> pairs, map<int, vector<t_DistDesc> > distances, vector <double *> polynomialsX, vector <double *> polynomialsY) {
    u = v = 0;
    double d2 = 0;
    double sumR = 0;
    for (int i = 0; i < pairs.size(); i++) {
        double rn = distances.at(i).at(distances.at(i).size() -1).dist;
        double a = p.x - pairs.at(i).p1.x;
        double b = p.y - pairs.at(i).p1.y;
        double r = sqrt(a * a + b * b + d2);

        

        double pX = 0;
        double pY = 0;
        for (int j = 0; j < 3; j++) {
            pX += polynomialsX.at(i)[j];
            pY += polynomialsY.at(i)[j];
        }
        if (r >= rn) {
            double R = 1 - 3 * r * r + 2 * r * r * r;    // 1 - 3r² + 2r³
            sumR += R;
            u += pX * R;
            v += pY * R;
        }
    }
    //u /= sumR;
    //v /= sumR;
}

inline void CRealMatches::getValueByCoefsWM(CvPoint2D32f p, double &u, double &v, vector<t_Pair> pairs, map<int, vector<t_DistDesc> > distances, double * coefs1, double * coefs2, int nMax) {
    //vector<t_DistDesc> points;
    //getDistancesVector(p, pairs, points, nMax);

    u = v = 0;
    double d2 = 0;
    double sumR = 0;
    //double rn = points.at(points.size() - 1).dist;
    for (int i = 0; i < pairs.size(); i++) {
        double a = p.x - pairs.at(i).p1.x;
        double b = p.y - pairs.at(i).p1.y;
        double r = 1/sqrt(a * a + b * b + d2);
        sumR += r;
        u += coefs1[i] * r;
        v += coefs2[i] * r;
    }
    u /= sumR;
    v /= sumR;
}

inline void CRealMatches::getValueByCoefsMQ(CvPoint2D32f p, double &u, double &v, vector<t_Pair> pairs, double * coefs1, double * coefs2) {
    u = v = 0;
    double d2 = 0;    
    for (int i = 0; i < pairs.size(); i++) {
        double a = p.x - pairs.at(i).p1.x;
        double b = p.y - pairs.at(i).p1.y;
        double r = sqrt(a * a + b * b + d2);
        u += coefs1[i] * r;
        v += coefs2[i] * r;        
    }
}