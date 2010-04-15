#include "CRealMatches.h"

void CRealMatches::changeDetection(IplImage * oldImg, IplImage * newImg) {
    // Select parameters for Gaussian model.
    CvGaussBGStatModelParams* params = new CvGaussBGStatModelParams;
    params->win_size = 2;
    params->n_gauss = 5;
    params->bg_threshold = 0.3;
    params->std_threshold = 3.5;
    params->minArea = 15;
    params->weight_init = 0.05;
    params->variance_init = 30;

    CvBGStatModel* bgModel = cvCreateGaussianBGModel(oldImg, params);

    cvUpdateBGStatModel(newImg, bgModel);

    // Display results
    cvNamedWindow("BG", 1);
    cvNamedWindow("FG", 1);
    cvShowImage("BG", bgModel->background);
    cvShowImage("FG", bgModel->foreground);

    cvReleaseBGStatModel(&bgModel);
}

// Nota: el problema aparece cuando la exponencial es negativa

inline double CRealMatches::normal(double x, double mean, double sdv) {
    return 1 / (sqrt(2 * CV_PI * sdv) * exp(((x - mean) * (x - mean)) / (2 * sdv)));
}

// Using Chauvenet's criterion: http://en.wikipedia.org/wiki/Chauvenet's_criterion
void CRealMatches::obstacleDetectionChauvenet(IplImage * pcaResult, IplImage * maskIn) {
    IplImage * obstaclesMask = cvCreateImage(size, IPL_DEPTH_8U, 1);
    IplImage * mask = cvCreateImage(size, IPL_DEPTH_8U, 1);
    cvCopyImage(maskIn, mask);

    CvScalar meanSc, stdDevSc;
    cvAvgSdv(pcaResult, &meanSc, &stdDevSc, mask);
    double mean = meanSc.val[0];
    double sdv = stdDevSc.val[0];
    int numMuestras = cvCountNonZero(mask);

    cvZero(obstaclesMask);
    for (int i = 0; i < pcaResult->height; i++) {
        for (int j = 0; j < pcaResult->width; j++) {
            double status = cvGetReal2D(mask, i, j);
            if (status != 255) continue;

            double val = cvGetReal2D(pcaResult, i, j);

            if (abs(val - mean) > 2 * sdv) {
                double prob = normal(val, mean, sdv);
                if (prob * numMuestras < 0.5) {
                    cvSetReal2D(obstaclesMask, i, j, 255);
                    cvSetReal2D(mask, i, j, 0);
                    numMuestras--;
                    cvAvgSdv(pcaResult, &meanSc, &stdDevSc, mask);
                    mean = meanSc.val[0];
                    sdv = stdDevSc.val[0];
                }
            }
        }
    }

    cvNamedWindow("ObstacleD", 1);
    cvShowImage("ObstacleD", obstaclesMask);

    /*cvErode(obstaclesMask, obstaclesMask);

    cvNamedWindow("ObstacleDEroded", 1);
    cvShowImage("ObstacleDEroded", obstaclesMask);

    cvDilate(obstaclesMask, obstaclesMask);

    cvNamedWindow("ObstacleDDilated", 1);
    cvShowImage("ObstacleDDilated", obstaclesMask);//*/

    cvReleaseImage(&obstaclesMask);
    cvReleaseImage(&mask);
}

void CRealMatches::obstacleDetectionQuartile(IplImage * pcaResult, IplImage * mask) {

    IplImage * obstaclesMask = cvCreateImage(size, IPL_DEPTH_8U, 1);

    double k = 2;

    // Ordenamos los datos
    vector<double> data;
    int levels[255];
    for (int i = 0; i < 255; i++) {
        levels[i] = 0;
    }

    for (int i = 0; i < pcaResult->height; i++) {
        for (int j = 0; j < pcaResult->width; j++) {
            if (cvGetReal2D(mask, i, j) != 255) continue;
            
            double val = cvGetReal2D(pcaResult, i, j);

            levels[(int)val]++;
        }
    }
    for (int i = 0; i < 255; i++) {
        for (int j = 0; j < levels[i]; j++) {
            data.push_back(i);
        }
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

    double maxThresh = q2 + k * (q2 - q1);

    cvZero(obstaclesMask);
    cvCopy(pcaResult, obstaclesMask, mask);
    cvThreshold(obstaclesMask, obstaclesMask, maxThresh, 255, CV_THRESH_BINARY);

    cvNamedWindow("ObstacleQ", 1);
    cvShowImage("ObstacleQ", obstaclesMask);    

    cvErode(obstaclesMask, obstaclesMask);

    //cvNamedWindow("ObstacleQEroded", 1);
    //cvShowImage("ObstacleQEroded", obstaclesMask);

    cvDilate(obstaclesMask, obstaclesMask);

    cvNamedWindow("ObstacleQDilated", 1);
    cvShowImage("ObstacleQDilated", obstaclesMask);//*/

    detectObstacles(obstaclesMask);

    cvReleaseImage(&obstaclesMask);
}

void CRealMatches::detectObstacles(IplImage * mask) {
    // Ahora buscamos contornos
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* contour = 0;
    IplImage * maskResult = cvCreateImage(size, IPL_DEPTH_8U, 3);
    IplImage * recuadro = cvCreateImage(size, IPL_DEPTH_8U, 3);
    cvCvtColor(img1, recuadro, CV_GRAY2BGR);

    cvFindContours(mask, storage, &contour, sizeof (CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    cvZero(maskResult);

    for (; contour != 0; contour = contour->h_next) {
        CvScalar color = CV_RGB(rand()&255, rand()&255, rand()&255);
        if (fabs(cvContourArea(contour)) > 50) {
            /* replace CV_FILLED with 1 to see the outlines */
            cvDrawContours(maskResult, contour, color, color, -1, CV_FILLED, 8);

            /*CvPoint * puntos = new CvPoint[contour->total];
            for (int i = 0; i < contour->total; i++) {
                    CvPoint * p = CV_GET_SEQ_ELEM(CvPoint, contour, i);
                    puntos[i] = *p;
                    cout << puntos[i].x << ", " << puntos[i].y << endl;
            }*/
            CvPoint p[4];
            p[0] = cvPoint(10, 10);
            p[1] = cvPoint(20, 10);
            p[2] = cvPoint(20, 20);
            p[3] = cvPoint(10, 20);
            CvRect rect = cvBoundingRect(contour);
            cvRectangle(recuadro, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width, rect.y + rect.height), cvScalar(0, 255, 0));
        }
    }

    cvNamedWindow("ResultadoPCA", 1);
    cvShowImage("ResultadoPCA", maskResult);

    cvNamedWindow("ResultadoFinal", 1);
    cvShowImage("ResultadoFinal", recuadro);

    cvReleaseImage(&recuadro);
    cvReleaseImage(&maskResult);
    cvReleaseMemStorage(&storage);
    delete contour;
}