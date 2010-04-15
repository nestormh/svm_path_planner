#include "../ImageRegistration.h"
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

int werwe( int argc, char** argv )
{
	assert( argc == 2 );
	IplImage* img = cvLoadImage( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	IplImage* rsp = cvLoadImage( argv[1], CV_LOAD_IMAGE_COLOR );
	CvSeq* contours;
	CvMemStorage* storage= cvCreateMemStorage();
	IplImage* hsv = cvCreateImage( cvGetSize( rsp ), IPL_DEPTH_8U, 3 );
	cvCvtColor( rsp, hsv, CV_BGR2YCrCb );
	double t = cvGetTickCount();
	cvExtractMSER( hsv, NULL, &contours, storage, cvMSERParams( 5, 60, cvRound(.2*img->width*img->height), .25, .2 ) );
	t = cvGetTickCount() - t;
	printf( "MSER extracted %d in %g ms.\n", contours->total, t/((double)cvGetTickFrequency()*1000.) );
	uchar* rsptr = (uchar*)rsp->imageData;
	// draw mser with different color
	for ( int i = contours->total-1; i >= 0; i-- )
	{
		CvSeq* r = *(CvSeq**)cvGetSeqElem( contours, i );
		for ( int j = 0; j < r->total; j++ )
		{
			CvPoint* pt = CV_GET_SEQ_ELEM( CvPoint, r, j );
			rsptr[pt->x*3+pt->y*rsp->widthStep] = bcolors[i%9][2];
			rsptr[pt->x*3+1+pt->y*rsp->widthStep] = bcolors[i%9][1];
			rsptr[pt->x*3+2+pt->y*rsp->widthStep] = bcolors[i%9][0];
		}
	}
	// find ellipse ( it seems cvfitellipse2 have error or sth?
	for ( int i = 0; i < contours->total; i++ )
	{
		CvContour* r = *(CvContour**)cvGetSeqElem( contours, i );
		CvBox2D box = cvFitEllipse2( r );
		/*
		if ( r->color > 0 )
			cvEllipseBox( rsp, box, colors[9], 2 );
		else
			cvEllipseBox( rsp, box, colors[10], 2 );
			*/
	}

	cvSaveImage( "rsp.png", rsp );

	cvNamedWindow( "original", 1 );
	cvShowImage( "original", img );
	
	cvNamedWindow( "response", 1 );
	cvShowImage( "response", rsp );

	cvWaitKey(0);

	cvDestroyWindow( "original" );
	cvDestroyWindow( "response" );
	
}
