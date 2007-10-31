#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>

#include "colorSegmentation.h"


CvMat** tSMeans;
CvMat* tSMasses;
CvMat** tSCovar;

CvMat** lSMeans;
CvMat* lSMasses;
CvMat** lSCovar;

void initPatterns(int n,int k) {
	lSMasses = cvCreateMat(n,1,CV_32FC1);
	tSMasses = cvCreateMat(n,1,CV_32FC1);
	cvZero(lSMasses);
	cvZero(tSMasses);
	
	lSCovar = (CvMat**)malloc(sizeof(CvMat*)*n);
	lSMeans = (CvMat**)malloc(sizeof(CvMat*)*n);

	tSCovar = (CvMat**)malloc(sizeof(CvMat*)*k);
	tSMeans = (CvMat**)malloc(sizeof(CvMat*)*k);

	for(int i = 0; i < n; i++){

		
		tSCovar[i] = cvCreateMat(3,3,CV_32FC1);
		cvZero(tSCovar[i]);
		tSMeans[i] = cvCreateMat(3,1,CV_32FC1);
		cvZero(tSMeans[i]);
		lSCovar[i] = cvCreateMat(3,3,CV_32FC1);
		cvZero(lSCovar[i]);
		lSMeans[i] = cvCreateMat(3,1,CV_32FC1);
		cvZero(lSMeans[i]);
	}
}
void kMeansClustering (IplImage* src, int k) {
	
	int h = 100,w = 200;
	float trainingSize = h*w;
	float pixels[MAX_TRAINING_PIXELS][3];

	
	IplImage* rgb = cvCreateImage(cvGetSize(src),IPL_DEPTH_32F,3); rgb->origin = 1;
	IplImage* auxImage = cvCreateImage(cvGetSize(src),IPL_DEPTH_32F,3); auxImage->origin = 1;
	
	
	
	

	CvMat* trainingAux = cvCreateMat(h,w,CV_32FC3);
	CvMat* clusters = cvCreateMat(trainingSize,1,CV_32SC1);
	CvMat* clusterMask = cvCreateMat(trainingSize,1,CV_8UC1);
	CvMat* training = cvCreateMat(trainingSize,1,CV_32FC3);
	CvMat* clusterIndex = cvCreateMat(k,1,CV_32SC1);
	
	CvMat rgbPixels[MAX_TRAINING_PIXELS];
	
	

	cvConvertScale(src,rgb);

	

	cvSet(auxImage,cvScalar(255.0,255.0,255.0));
	cvDiv(rgb,auxImage,rgb);
	cvGetSubRect(rgb,trainingAux,cvRect(50,0,w,h));
	cvRepeat(trainingAux,training);

	

	cvKMeans2(training,k,clusters,cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0 ));
	
	cvZero(clusterIndex);
	for (int i = 0; i <k ; i++) {

		

		cvCmpS(clusters,i,clusterMask,CV_CMP_EQ);
		tSMasses->data.fl[i] = cvCountNonZero(clusterMask);
		CvMat* clusterPixels = cvCreateMat(tSMasses->data.fl[i],1,CV_32FC3);
		
		
		
		CvMat *vects[MAX_TRAINING_PIXELS] = {0};

		
		for (int j = 0; j < trainingSize;j++) {
			if (clusters->data.i[j] == i) {
				CvScalar pix;
				pix = cvGet1D(training,j);
				cvSet1D(clusterPixels,clusterIndex->data.i[i]++,pix);
			}		
		}
		
		for (int j = 0; j < tSMasses->data.fl[i];j++) {
			
			CvScalar auxPix;

			auxPix = cvGet1D(clusterPixels,j);

			pixels[j][0] = auxPix.val[0];
			pixels[j][1] = auxPix.val[1];
			pixels[j][2] = auxPix.val[2];
			
			rgbPixels[j] = cvMat(3,1,CV_32FC1,pixels[j]);
			
			
			vects[j] = &rgbPixels[j];
			//printf ("%d %f %f %f\n", j,vects[j]->data.fl[0],vects[j]->data.fl[1],vects[j]->data.fl[2]);
		}

		cvReleaseMat(&clusterPixels);
		//cvZero(tSCovar[i]);
		//float mat0[9];
		//CvMat cvMat0 = cvMat( 3, 3, CV_32FC1, mat0 );   
		//for(int w = 0; w < tSMasses->data.fl[i];w++)
			//printf ("%d %f %f %f\n", w,vects[w]->data.fl[0],vects[w]->data.fl[1],vects[w]->data.fl[2]);
			//printf ("Cluster %d %d %f %f %f\n", i,w,rgbPixels[w].data.fl[0],rgbPixels[w].data.fl[1],rgbPixels[w].data.fl[2]);
		cvCalcCovarMatrix((const void**)&vects[0],(int)floor(tSMasses->data.fl[i]),tSCovar[i],tSMeans[i],CV_COVAR_SCALE|CV_COVAR_NORMAL);
		//printf ("covar----->: %d\n",cvCountNonZero(&cvMat0)); 
		
		
		
		
	}
	
	cvRectangle(src,cvPoint(50,0),cvPoint(w,h),cvScalar(255));
	cvNamedWindow( "Source", 2 );
    cvShowImage( "Source", src );
	
	cvReleaseImage(&rgb);
	
	cvReleaseMat(&training);
	cvReleaseMat(&trainingAux);
	cvReleaseMat(&clusters);
	cvReleaseMat(&clusterMask);
	cvReleaseMat(&clusterIndex);
	
}

void mostrarPatrones(int k, IplImage* img) {

	float r,g,b;
//	printf ("*******************************************\n"); 
	for (int i = 0; i < k; i++) {
		r = lSMeans[i]->data.fl[0]*255;
		g = lSMeans[i]->data.fl[1]*255;
		b = lSMeans[i]->data.fl[2]*255;
		
		int w = img->width;
		int base = (int)floor((float)w/(float)k);
	

		cvRectangle(img,cvPoint(i*base,0),cvPoint((i+1)*base-1,200),cvScalar(b,g,r),CV_FILLED);
		
		printf ("Cluser %d:\nMedia-> r:%f g:%f b:%f\n",i, r,g,b);
		printf ("Covar:\n"); 
		for (int j = 0; j < 3; j++) {
			for (int w = 0; w < 3; w++) {
				CvScalar val = cvGet2D(lSCovar[i],j,w);
				printf ("%f ",val.val[0]);
			}
			printf("\n");
		}
		
	}
	cvNamedWindow( "Patrones", 3 );
    cvShowImage( "Patrones", img );
}

void patronesAprendizaje(int k, int n) {

	IplImage* paletaPatrones = cvCreateImage(cvSize(320,240),8,3); paletaPatrones->origin = 1;
	CvMat *aux,*aux1,*aux2,*aux3,*aux4,*aux5,*aux6;
	
	aux = cvCreateMat(3,3,CV_32FC1);
	aux2 = cvCreateMat(3,3,CV_32FC1);

	aux3 = cvCreateMat(3,1,CV_32FC1);
	aux4 = cvCreateMat(3,1,CV_32FC1);

	aux5 = cvCreateMat(1,3,CV_32FC1);

	aux6 = cvCreateMat(1,1,CV_32FC1);



	for (int i = 0; i < k; i++) {
		bool match = false;
		for (int j = 0; j < n; j++) {	
			if (lSMasses->data.fl[j] != 0) {
				cvSub(lSMeans[j],tSMeans[i],aux3);
				cvAdd(lSCovar[j],tSCovar[i],aux);
				cvInvert(aux,aux,CV_LU);
				cvGEMM(aux3,aux,1,0,0,aux5,CV_GEMM_A_T);
				cvGEMM(aux5,aux3,1,0,0,aux6,0);
				//printf ("%d %d\n%d %d\n", aux5->rows, aux5->cols,aux->rows,aux->cols);
				//cvMul(aux5,aux,aux4);
				//cvMul(aux5,aux3,aux6);
				CvScalar val = cvGet1D(aux6,0);
				//printf ("%f\n", val.val[0]);
				if(val.val[0] <= 1) {
					match = true;
					
					cvSet(aux3,cvScalar(lSMasses->data.fl[j]));
					cvMul(aux3,lSMeans[j],aux3);
					cvSet(aux4,cvScalar(tSMasses->data.fl[i]));
					cvMul(aux4,tSMeans[i],aux4);
					cvAdd(aux3,aux4,aux3);
					cvSet(aux4,cvScalar(lSMasses->data.fl[j]+tSMasses->data.fl[i]));
					cvDiv(aux3,aux4,aux3);
					cvCopy(aux3,lSMeans[j]);
					
					cvSet(aux,cvScalar(lSMasses->data.fl[j]));
					cvMul(aux,lSCovar[j],aux);
					cvSet(aux2,cvScalar(tSMasses->data.fl[i]));
					cvMul(aux2,tSCovar[i],aux2);
					cvAdd(aux,aux2,aux);
					cvSet(aux2,cvScalar(lSMasses->data.fl[j]+tSMasses->data.fl[i]));
					cvDiv(aux,aux2,aux);
					cvCopy(aux,lSCovar[j]);
					
					if (lSMasses->data.fl[j] < 100000)
						lSMasses->data.fl[j] = lSMasses->data.fl[j]+tSMasses->data.fl[i];
					
				}
			}
		}
		if (!match) {
			int ind = 0, mass = lSMasses->data.fl[0];

			for (int j = 0; j < n; j++) {
				if (lSMasses->data.fl[j] < mass){
					mass = lSMasses->data.fl[j];
					ind = j;
				}
			}
			cvCopy(tSMeans[i],lSMeans[ind]);
			cvCopy(tSCovar[i],lSCovar[ind]);
			lSMasses->data.fl[ind] = tSMasses->data.fl[i];
		}
	}

	cvZero(paletaPatrones);
	mostrarPatrones(n,paletaPatrones);
cvReleaseImage(&paletaPatrones);
cvReleaseMat(&aux);
cvReleaseMat(&aux2);
cvReleaseMat(&aux3);
cvReleaseMat(&aux4);
cvReleaseMat(&aux5);
cvReleaseMat(&aux6);
}
void mahalanobisDistance(IplImage* img,int n) {
	IplImage* mahalanobis = cvCreateImage(cvGetSize(img),8,1);mahalanobis->origin = 1;
	
	double min,max;
	CvMat* mah = cvCreateMat(img->height,img->width,CV_32FC1);
	cvZero(mahalanobis);
	float mass = 0;
	 void* vects[2];
					 float data[9];
					 float data2[3];
	for (int i = 0; i < n;i++)
		if (mass > lSMasses->data.fl[i])
			mass = lSMasses->data.fl[i];
	for (int i = 0; i < img->width; i++) {
		for (int j = 0; j < img->height;j++) {
			float pixel[3];
			CvScalar val = cvGet2D(img,j,i);
			pixel[0] = val.val[0]/255.0;
			pixel[1] = val.val[1]/255.0;
			pixel[2] = val.val[2]/255.0;
			

			CvMat vec1 = cvMat(3,1,CV_32FC1,pixel);
			float m = 1000000;
			for (int u = 0; u < n; u++) {
				if (lSMasses->data.fl[u] > 0.33*mass) {
					
					 CvMat covar = cvMat(3,3,CV_32FC1,data);
					 /*
					 CvMat mean = cvMat(3,1,CV_32FC1,data2);
					 vects[0] = &vec1;
					 vects[1] = lSMeans[u];
					 cvCalcCovarMatrix((const void **) vects,2,&covar,&mean,CV_COVAR_SCALE|CV_COVAR_NORMAL);
					 */
					 cvInvert(lSCovar[u],&covar);
					 float aux = cvMahalanobis(&vec1,lSMeans[u],&covar);
					 if (aux < m)
						 m = aux;
				}
			}
			//int aux = (int)floor(255.0*((float)1.0/(float)m));
			//printf ("%f\n", m);
			cvSet2D(mah,j,i,cvScalar(m));
			
			
		}
	}

	cvMinMaxLoc(mah,&min,&max);
	for (int i = 0; i < img->width; i++) {
		for (int j = 0; j < img->height;j++) {
			CvScalar val = cvGet2D(mah,j,i);
			//printf ("%f %f %f\n",val.val[0],min,max);
			if (val.val[0] <= 1) val.val[0] = 1;
			else val.val[0] = 1.0/val.val[0];
			CV_IMAGE_ELEM(mahalanobis,unsigned char,j,i) = (unsigned char)floor(255.0*val.val[0]);
		}
	}
	//cvEqualizeHist(mahalanobis,mahalanobis);
	//cvThreshold(mahalanobis,mahalanobis,50,255,CV_THRESH_BINARY);
	cvNamedWindow( "Mahalanobis", 4 );
	cvShowImage("Mahalanobis",mahalanobis);
	cvReleaseImage(&mahalanobis);
}