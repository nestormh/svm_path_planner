#include "stdafx.h"
#include "NavEntorno.h"


using namespace std;

void initGUI() {
	cvNamedWindow(W_PERSP, 1);
	cvMoveWindow(W_PERSP, 0, 500);
	cvNamedWindow(W_FLOW2, 1);	
	cvMoveWindow(W_FLOW2, 0, 250);
	cvNamedWindow(W_FLOW1, 1);
	cvMoveWindow(W_FLOW1, 0, 0);
	
	cvNamedWindow(W_PLOT_PCA, 1);
	cvMoveWindow(W_PLOT_PCA, 1005, 290);
	cvNamedWindow(W_PLOT, 1);
	cvMoveWindow(W_PLOT, 1005, 0);
	
	cvNamedWindow(W_RESTA, 1);
	cvMoveWindow(W_RESTA, 670, 500);
	cvNamedWindow(W_VEL_OFLOW, 1);
	cvMoveWindow(W_VEL_OFLOW, 670, 250);
	cvNamedWindow(W_DISTPCA, 1);
	cvMoveWindow(W_DISTPCA, 670, 0);

	cvNamedWindow("Puntos", 1);
	cvMoveWindow("Puntos", 335, 0);	
}
/**
// Constructor: Sets initial value of variables (actually NULL)
*/
CNavEntorno::CNavEntorno(void) {
	velx = NULL;
	vely = NULL;	
	img1 = NULL;
	img2 = NULL;
	A = NULL;
	B = NULL;
	X = NULL;
	warp = NULL;
	persp = NULL;
	perspMask = NULL;
	imgShaped = NULL;
	data = NULL;
	data1 = NULL;
	data2 = NULL;
	pcaData = NULL;
	corr = NULL;
	avg = NULL;
	eigenValues = NULL;
	eigenVectors =  NULL;
	dataX = NULL;
	dataY = NULL;
	distPCA = NULL;
	vel = NULL;
	subImages = NULL;
}

/**
// Destructor: Checks if variables have been used. If so, it frees used memory
*/
CNavEntorno::~CNavEntorno(void) {
	if (velx != NULL) cvReleaseMat(&velx);
	if (vely != NULL) cvReleaseMat(&vely);
	if (A != NULL) cvReleaseMat(&A);
	if (B != NULL) cvReleaseMat(&B);
	if (X != NULL) cvReleaseMat(&X);
	if (warp != NULL) cvReleaseMat(&warp);
	if (persp != NULL) cvReleaseImage(&persp);
	if (perspMask != NULL) cvReleaseImage(&perspMask);
	if (imgShaped != NULL) cvReleaseImage(&imgShaped);
	if (data != NULL) cvReleaseMat(&data);
	if (data1 != NULL) cvReleaseMat(&data1);
	if (data2 != NULL) cvReleaseMat(&data2);
	if (pcaData != NULL) cvReleaseMat(&pcaData);
	if (corr != NULL) cvReleaseMat(&corr);
	if (avg != NULL) cvReleaseMat(&avg);
	if (eigenValues != NULL) cvReleaseMat(&eigenValues);
	if (eigenVectors != NULL) cvReleaseMat(&eigenVectors);
	if (dataX != NULL) cvReleaseMat(&dataX);
	if (dataY != NULL) cvReleaseMat(&dataY);
	if (distPCA != NULL) cvReleaseMat(&distPCA);
	if (vel != NULL) cvReleaseMat(&vel);
	if (subImages != NULL) cvReleaseImage(&subImages);
}

/**
//	Initializes the memory in the first iteration
//	@param img	The first image in the first iteration. It gives information about width and height
*/
inline void CNavEntorno::init(IplImage * img) {
	if (velx == NULL || vely == NULL ||
		A == NULL || B == NULL || X == NULL || warp == NULL ||
		persp == NULL || perspMask == NULL || imgShaped == NULL ||
		data == NULL || data1 == NULL || data2 == NULL || pcaData == NULL||
		dataX == NULL || dataY == NULL || distPCA == NULL || vel == NULL ||
		subImages == NULL) {		// If variables have been initializes, it does nothing

		if (velx != NULL) cvReleaseMat(&velx);	// If velx had been initialized, it will change (it wouldn't be executed)
		if (vely != NULL) cvReleaseMat(&vely);	// If vely had been initialized, it will change (it wouldn't be executed)
		if (A != NULL) cvReleaseMat(&A);	// If A had been initialized, it will change (it wouldn't be executed)
		if (B != NULL) cvReleaseMat(&B);	// If B had been initialized, it will change (it wouldn't be executed)
		if (X != NULL) cvReleaseMat(&X);	// If X had been initialized, it will change (it wouldn't be executed)
		if (warp != NULL) cvReleaseMat(&warp);	// If warp had been initialized, it will change (it wouldn't be executed)
		if (persp != NULL) cvReleaseImage(&persp);	// If persp had been initialized, it will change (it wouldn't be executed)
		if (perspMask != NULL) cvReleaseImage(&perspMask);	// If perspMask had been initialized, it will change (it wouldn't be executed)
		if (imgShaped != NULL) cvReleaseImage(&imgShaped);	// If imgShaped had been initialized, it will change (it wouldn't be executed)
		if (data != NULL) cvReleaseMat(&data);	// If data had been initialized, it will change (it wouldn't be executed)
		if (data1 != NULL) cvReleaseMat(&data1);	// If data1 had been initialized, it will change (it wouldn't be executed)
		if (data2 != NULL) cvReleaseMat(&data2);	// If data2 had been initialized, it will change (it wouldn't be executed)
		if (pcaData != NULL) cvReleaseMat(&pcaData);	// If pcaData had been initialized, it will change (it wouldn't be executed)
		if (corr != NULL) cvReleaseMat(&corr);	// If corr had been initialized, it will change (it wouldn't be executed)
		if (avg != NULL) cvReleaseMat(&avg);	// If avg had been initialized, it will change (it wouldn't be executed)
		if (eigenValues != NULL) cvReleaseMat(&eigenValues);	// If eigenValues had been initialized, it will change (it wouldn't be executed)
		if (eigenVectors != NULL) cvReleaseMat(&eigenVectors);	// If eigenVectors had been initialized, it will change (it wouldn't be executed)
		if (dataX != NULL) cvReleaseMat(&dataX);	// If dataX had been initialized, it will change (it wouldn't be executed)
		if (dataY != NULL) cvReleaseMat(&dataY);	// If dataY had been initialized, it will change (it wouldn't be executed)
		if (distPCA != NULL) cvReleaseMat(&distPCA);	// If distPCA had been initialized, it will change (it wouldn't be executed)
		if (vel != NULL) cvReleaseMat(&vel);	// If vel had been initialized, it will change (it wouldn't be executed)
		if (subImages != NULL) cvReleaseImage(&subImages);	// If subImage had been initialized, it will change (it wouldn't be executed)
	
		velx = cvCreateMat(img->height, img->width, CV_32FC1);
		vely = cvCreateMat(img->height, img->width, CV_32FC1);
						
		int size = (img1->width * img1->height / step) * 2;
		A = cvCreateMat(size, 8, CV_64FC1);
		cvSet(A, cvScalar(0));
		B = cvCreateMat(size, 1, CV_64FC1);
		X = cvCreateMat(8, 1, CV_64FC1);
		warp = cvCreateMat(3, 3, CV_64FC1);

		persp = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

		perspMask = cvCreateImage(cvGetSize(persp), IPL_DEPTH_8U, 1);
		imgShaped = cvCreateImage(cvGetSize(persp), IPL_DEPTH_8U, 1);

		int length = img->width * img->height;

		data = cvCreateMat(2, length, CV_64FC1);
		data1 = cvCreateMat(1, length, CV_64FC1);
		data2 = cvCreateMat(1, length, CV_64FC1);		
		pcaData = cvCreateMat(2, length, CV_64FC1);
		dataX = cvCreateMat(1, img->width * img->height, CV_64FC1);
		dataY = cvCreateMat(1, img->width * img->height, CV_64FC1);
		distPCA = cvCreateMat(img->height, img->width, CV_64FC1);

		data1 = cvGetRow(data, data1, 0);
		data2 = cvGetRow(data, data2, 1);		

		corr = cvCreateMat(2, 2, CV_64FC1);
		avg = cvCreateMat(1, 2, CV_64FC1);
		eigenValues = cvCreateMat(1, 2, CV_64FC1);
		eigenVectors = cvCreateMat(2, 2, CV_64FC1);

		vel = cvCreateMat(img1->height, img1->width, CV_32FC1);

		subImages = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
	}
}

/**
//	Transforms the base image
*/
inline void CNavEntorno::warpImage() {	
	int cuenta = img1->width * img1->height / step;
							
	for (int i = 0, pos = 0, pos2 = cuenta; i < img1->width; i += step) {
		for (int j = 0; j < img1->height; j += step, pos++, pos2++) {			
			double dx = i;
			//if (cvGetReal2D(velx, j, i) <= 20)
				dx -= cvGetReal2D(velx, j, i);
			double dy = j;
			//if (cvGetReal2D(vely, j, i) <= 20)
				dy -= cvGetReal2D(vely, j, i);

			cvSetReal2D(A, pos, 0, i);
			cvSetReal2D(A, pos2, 3, i);
			cvSetReal2D(A, pos, 1, j);
			cvSetReal2D(A, pos2, 4, j);
			cvSetReal2D(A, pos, 2, 1);
			cvSetReal2D(A, pos2, 5, 1);
			// DUDA_EXISTENCIAL
			cvSetReal2D(A, pos, 6, -i*dx);
			cvSetReal2D(A, pos2, 6, -i*dy);
			cvSetReal2D(A, pos, 7, -j*dx);
			cvSetReal2D(A, pos2, 7, -j*dy);			
			// FIN DE DUDA_EXISTENCIAL

			cvSetReal2D(B, pos, 0, dx);
			cvSetReal2D(B, pos2, 0, dy);					
		}
	}
		
	cvSolve(A, B, X, CV_SVD);		
	
	for (int i = 0, pos = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++, pos++) {
			if (pos != 8)
				cvSetReal2D(warp, i, j, cvGetReal1D(X, pos));
			else
				cvSetReal2D(warp, i, j, 1.0);
		}
	}

	cvWarpPerspective(img1, persp, warp);
}


/**
//	Calculates PCA for persp and img2
*/
inline void CNavEntorno::calcPCA() {	
	cvThreshold(persp, perspMask, 0, 255, CV_THRESH_BINARY);
	cvZero(imgShaped);
	cvCopy(img2, imgShaped, perspMask);
			
	for (int i = 0, pos = 0; i < persp->width; i++)	{
		for (int j = 0; j < persp->height; j++, pos++) {
			cvmSet(data, 0, pos, cvGetReal2D(persp, j, i));
			cvmSet(data, 1, pos, cvGetReal2D(imgShaped, j, i));
		}
	}	

	//cvSmooth(persp, persp);
	//cvSmooth(img2, img2);
	
	double m1 = cvMean(data1);
	double m2 = cvMean(data2);	

	cvSubS(data1, cvScalar(m1), data1);
	cvSubS(data2, cvScalar(m2), data2);

	cvMulTransposed(data, corr, 0);	

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {		
			cvmSet(corr, i, j, cvmGet(corr, i, j) / (1 - persp->width * persp->height));			
		}		
	}

	cvCalcPCA(corr, avg, eigenValues, eigenVectors, CV_PCA_DATA_AS_ROW);

	double a = cvmGet(corr, 0, 0);
	double b = cvmGet(corr, 0, 1);
	double c = cvmGet(corr, 1, 0);
	double d = cvmGet(corr, 1, 1);

	double m = (-a) - d;
	double n = a * d - b * c;

	double lambda1 = (-m + sqrt(pow(m, 2.0) - 4 * n)) / 2.0;
	double lambda2 = (-m - sqrt(pow(m, 2.0) - 4 * n)) / 2.0;

	double e2 = sqrt(pow(c, 2.0) / (pow(-(d - lambda1), 2.0) + pow(c, 2.0)));
	double e1 = -(d - lambda1) * e2 / c;

	cvmSet(eigenVectors, 0, 0, e1);
	cvmSet(eigenVectors, 1, 0, e2);

	e2 = sqrt(pow(c, 2.0) / (pow(-(d - lambda2), 2.0) + pow(c, 2.0)));
	e1 = e1 = -(d - lambda2) * e2 / c;

	cvmSet(eigenVectors, 0, 1, e1);
	cvmSet(eigenVectors, 1, 1, e2);
	
	cvMatMul(eigenVectors, data, pcaData);	
}

inline void CNavEntorno::getDifsOnPCA() {
	// Gets ACP vectors X and Y
	dataX = cvGetRow(pcaData, dataX, 1);
	dataY = cvGetRow(pcaData, dataY, 0);		
	
	// Calculates mean and stdev
	cvAvgSdv(dataX, &xMean, &xSdv);
	cvAvgSdv(dataY, &yMean, &ySdv);	
		
	// Draws ACP graphic data	
	for (int i = 0, pos = 0; i < img1->width; i++) {
		for (int j = 0; j < img2->height; j++, pos++) {									
			cvSetReal2D(distPCA, j, i, abs(cvGetReal1D(dataY, pos) - yMean.val[0]));
		}
	}
}

inline void CNavEntorno::calcOFlowDistancesAndSub() {				
	cvMul(velx, velx, velx);
	cvMul(vely, vely, vely);
	cvAdd(velx, vely, vel);
	cvNormalize(vel, vel, 0, 255, CV_MINMAX);
	
	cvSub(img2, persp, subImages);	
}

inline void CNavEntorno::igualaBrilloContraste() {
	CvScalar media1, desv1;
	CvScalar media2, desv2;

	CvMat * mLut = cvCreateMatHeader(1, 256, CV_8UC1);	
	uchar lut[256];
	cvSetData(mLut, lut, 0);

	cvAvgSdv(img1, &media1, &desv1);
	cvAvgSdv(img2, &media2, &desv2);

	double brightness = ((media1.val[0] - media2.val[0]) * 100 / 128);
	double contrast = ((desv1.val[0] - desv2.val[0]) * 100 / 128);	
	contrast = 0;

	/*
     * The algorithm is by Werner D. Streidt
     * (http://visca.com/ffactory/archives/5-99/msg00021.html)
     */
    if( contrast > 0 )
    {
        double delta = 127.*contrast/100;
        double a = 255./(255. - delta*2);
        double b = a*(brightness - delta);
        for(int i = 0; i < 256; i++ )
        {
            int v = cvRound(a*i + b);
            if( v < 0 )
                v = 0;
            if( v > 255 )
                v = 255;
            lut[i] = (uchar)v;
        }
    }
    else
    {
        double delta = -128.*contrast/100;
        double a = (256.-delta*2)/255.;
        double b = a*brightness + delta;
        for(int i = 0; i < 256; i++ )
        {
            int v = cvRound(a*i + b);
            if( v < 0 )
                v = 0;
            if( v > 255 )
                v = 255;
			lut[i] = (uchar)v;           
        }
    }

    cvLUT(img1, img1, mLut);

	cvReleaseMat(&mLut);
}

inline void CNavEntorno::logicTransf() {				
	CvMat * mResta = cvCreateMat(img1->height, img1->width, CV_64FC1);
	CvMat * mVel = cvCreateMat(img1->height, img1->width, CV_64FC1);
	CvMat * mRestaAndVel = cvCreateMat(img1->height, img1->width, CV_64FC1);
	CvMat * mRestaAndPCA = cvCreateMat(img1->height, img1->width, CV_64FC1);
	CvMat * mVelAndPCA = cvCreateMat(img1->height, img1->width, CV_64FC1);
	CvMat * result = cvCreateMat(img1->height, img1->width, CV_64FC1);
	IplImage * imgResult = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
	
	cvNormalize(distPCA, distPCA, 0, 255, CV_MINMAX);
	cvNormalize(vel, mVel, 0, 255, CV_MINMAX);
	cvNormalize(subImages, mResta, 0, 255, CV_MINMAX);
	cvOr(mResta, mVel, mRestaAndVel);
	cvAdd(mResta, distPCA, mRestaAndPCA);
	cvOr(mVel, distPCA, mVelAndPCA);

	cvAnd(mRestaAndVel, mRestaAndPCA, result);
	cvAnd(result, mVelAndPCA, result);

	cvSubRS(mResta, cvScalar(255), mResta);
	cvSub(distPCA, mResta, mRestaAndPCA);
	cvNormalize(mRestaAndPCA, imgResult, 0, 255, CV_MINMAX);

	cvShowImage("Puntos", imgResult);

	cvReleaseMat(&mResta);
	cvReleaseMat(&mRestaAndVel);
	cvReleaseMat(&mRestaAndPCA);
	cvReleaseMat(&mVelAndPCA);	
	cvReleaseMat(&result);	
	cvReleaseImage(&imgResult);
}

/**
//	Shows execution information on the screen 
//	@param paint Tells if it would paint or not
//	@returns	The time expended in painting
*/
inline void CNavEntorno::showData(bool paint, double time, double absDist, double angDist, double lateralDist) {
	if (paint == false) {
		return;
	}	

	CvFont font;
	double hScale=0.6;
	double vScale=0.7;
	int lineWidth=2;
	cvInitFont(&font,CV_FONT_VECTOR0, hScale,vScale,0,lineWidth);
	
	char text[1024];

	IplImage * oFlow1 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);
	IplImage * oFlow2 = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);	
	IplImage * grafico = cvCreateImage(cvSize(256, 256), IPL_DEPTH_8U, 3);
	IplImage * graficoPre = cvCreateImage(cvSize(256, 256), IPL_DEPTH_8U, 3);
	CvMat * dataACP = cvCreateMat(2, pcaData->cols, CV_64FC1);	
	CvMat * myData = cvCreateMat(2, pcaData->cols, CV_64FC1);	
	IplImage * result = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
	IplImage * statistics = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 3);

	cvCvtColor(img1, oFlow1, CV_GRAY2BGR);
	cvCvtColor(img2, oFlow2, CV_GRAY2BGR);

	for (int i = 0; i < img1->width; i += step) {
		for (int j = 0; j < img1->height; j += step) {
			double dx = cvGetReal2D(velx, j, i);
			double dy = cvGetReal2D(vely, j, i);			

			cvCircle(oFlow1, cvPoint(i, j), 3, cvScalar(255, 0, 0), 1);
			cvCircle(oFlow2, cvPoint(i, j), 3, cvScalar(255, 0, 0), 1);
			cvCircle(oFlow2, cvPoint(i + dx, j + dy), 3, cvScalar(0, 0, 255), 1);
			cvLine(oFlow2, cvPoint(i, j), cvPoint(i + dx, j + dy), cvScalar(0, 255, 0), 1);			
		}
	}

	// Draws ACP graphic data
	cvNormalize(pcaData, dataACP, 0, 255, CV_MINMAX);
	cvNormalize(data, myData, 0, 255, CV_MINMAX);
	cvZero(grafico);
	cvZero(graficoPre);
	for (int i = 0, pos = 0; i < img1->width; i++) {
		for (int j = 0; j < img2->height; j++, pos++) {			
			cvSet2D(grafico, cvGetReal2D(dataACP, 0, pos), cvGetReal2D(dataACP, 1, pos), cvScalar(0, 255, 0));
			cvSet2D(graficoPre, cvGetReal2D(myData, 0, pos), cvGetReal2D(myData, 1, pos), cvScalar(0, 0, 255));
			if (cvGetReal2D(vel, j, i) > 20) {
				cvSetReal2D(vel, j, i, 20);
			}
		}
	}	

	// Shows Statistics
	cvZero(statistics);
	sprintf(text, "Time: %f milisec", time);
	cvPutText (statistics, text, cvPoint(10,30), &font, cvScalar(255,255,255));
	sprintf(text, "Dist Abs: %f m", absDist);
	cvPutText (statistics, text, cvPoint(10,60), &font, cvScalar(255,255,255));
	sprintf(text, "Dist Ang: %f �", angDist);
	cvPutText (statistics, text, cvPoint(10,90), &font, cvScalar(255,255,255));
	sprintf(text, "Lat dist: %f", lateralDist);
	cvPutText (statistics, text, cvPoint(10,120), &font, cvScalar(255,255,255));
	sprintf(text, "Mean: %f, %f", xMean.val[0], yMean.val[0]);
	cvPutText (statistics, text, cvPoint(10,150), &font, cvScalar(255,255,255));
	sprintf(text, "Sdv: %f, %f", xSdv.val[0], ySdv.val[0]);
	cvPutText (statistics, text, cvPoint(10,180), &font, cvScalar(255,255,255));

	cvShowImage(W_FLOW1, oFlow1);
	cvShowImage(W_FLOW2, oFlow2);	
	cvShowImage(W_PERSP, persp);	
	cvNormalize(subImages, result, 0, 255, CV_MINMAX);
	cvShowImage(W_RESTA, result);
	cvShowImage(W_PLOT_PCA, grafico);
	cvShowImage(W_PLOT, graficoPre);
	cvNormalize(distPCA, result, 0, 255, CV_MINMAX);
	cvShowImage(W_DISTPCA, result);
	cvNormalize(vel, result, 0, 255, CV_MINMAX);
	cvShowImage(W_VEL_OFLOW, result);
	cvShowImage("Control", statistics);

	logicTransf();

	cvReleaseImage(&oFlow1);
	cvReleaseImage(&oFlow2);	
	cvReleaseImage(&grafico);
	cvReleaseImage(&graficoPre);
	cvReleaseMat(&dataACP);
	cvReleaseMat(&myData);
	cvReleaseImage(&result);
	cvReleaseImage(&statistics);	
}

/**
//	Fix img1 to keep the maximum similarities against img1,
//	@param img1	Base image
//	@param img2	Real time image
//	@param persp Modified img1 image
//	@param time	Expended time in this method
//	@param paint Tells if it would paint results or not
*/
void CNavEntorno::matchImages(IplImage * img1, IplImage * img2, IplImage * persp, double &time, double absDist, double angDist, double lateralDist, bool paint) {
	this->img1 = img1;
	this->img2 = img2;
	init(img1);		

	clock_t	matchTime = clock();
	
	//igualaBrilloContraste();

	// First, we check the optical flow between the two images
	cvCalcOpticalFlowLK(img1, img2, winSize, velx, vely);
	warpImage();			

	calcPCA();

	getDifsOnPCA();
	calcOFlowDistancesAndSub();

	time = (double(clock() - matchTime) / CLOCKS_PER_SEC * 1000);

	showData(paint, time, absDist, angDist, lateralDist);	
}

IplImage * CNavEntorno::getDifPCA() {
	IplImage * ret = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
	cvNormalize(distPCA, ret, 0, 255, CV_MINMAX);
	return ret;
}

void CNavEntorno::getPCADifs(IplImage * persp, IplImage * img2, IplImage * mask, bool paint) {

	if (paint) initGUI();

	this->img1 = persp;
	this->img2 = img2;
	init(img1);

	this->persp = persp;
	this->perspMask = mask;
	
	calcPCA();
	getDifsOnPCA();

	if (paint)
		showData(true, 10, 10, 10, 10);

}