#include <cvaux.h>
#include <highgui.h>

#include "cvfast.h"

int main(int argc, char* argv[])
{
	////////////////////////////////////////////////////////////////////////////////
	//
	// Process command line arguments
	//
	
	int numArgs = argc;

	char* OutFileName = argv[numArgs-1];
	char* InFileName = argv[numArgs-2];

	int inFASTThreshhold = 20;
	int inNpixels = 9;
	int inModeCrossesVsList = 1;
	int inNonMaxSuppression  = 0;


	int numRemainingArgs = numArgs - 3;

	if( numArgs <=2 ) {
		printf( "Need at least 2 arguments: input_file output_file\n" );
		printf( "Re-run with /? to see usage\n" );
	}

	if( numArgs < 3 ) {
		printf( "Perform  FAST feature detection on an image.\n"
				"\n"
				"Usage:\n"
				"input_image \t Input filename\n"
				"output_file \t Output filename\n"
				"/l \t\t Output a list of corners (instead of an image)\n"
				"/t X \t\t Set threshold to X. Default is 20\n"
				"/n X \t\t Use X point fast (allowed range is 9--12, default is 9)\n"
				"/s \t\t Perform nonmaximal suppression\n");
		return 0;
	}

	for( int j=1; j <= numRemainingArgs; j++ )
	{
		if( !strcmp( argv[j], "/l" ) ) {
			inModeCrossesVsList = 2;
		}
		else if( !strcmp( argv[j], "/s" ) ) {
			inNonMaxSuppression = 1;
		}
		else if( !strcmp( argv[j], "/t" ) ) {
			j++;
			inFASTThreshhold = atoi( argv[j] );
		}
		else if( !strcmp( argv[j], "/n" ) ) {
			j++;
			inNpixels = atoi( argv[j] );
			if( inNpixels > 12 || inNpixels < 9 ) {
				printf( "X point fast requires 8 < X  < 13, not %d\n", inNpixels );
				return( 0 );
			}
		}
		else {
			printf( "Unrecognized arguments (%s), re-run with /?\n", argv[j] );
			return( 0 );
		}
	}

	////////////////////////////////////////////////////////////////////////////////
	//
	//Load an image in to 8 bit grey cvd image using IPL
	//
	
	IplImage* I_in = cvLoadImage( InFileName, -1 );
	IplImage* IGray = cvCreateImage( cvGetSize( I_in ), IPL_DEPTH_8U, 1);
	cvCvtColor( I_in, IGray, CV_RGB2GRAY );

	CvPoint* corners;
	int numCorners;
	
	cvCornerFast(IGray, inFASTThreshhold, inNpixels, inNonMaxSuppression, &numCorners, & corners);

	
	if ( inModeCrossesVsList == 1 ) 
	{
		// Put they grey image in to a colour buffer and draw on it
		cvCvtColor( IGray, I_in, CV_GRAY2RGB );	// Instead of iplColorToGray( Icanvas, IcanvasGray );
		for(int i=0; i < numCorners; i++ ) 
		{
			cvLine( I_in, 
					cvPoint( corners[i].x-1, corners[i].y ), 
					cvPoint( corners[i].x+1, corners[i].y ), 
					CV_RGB(255,0,0) );
			cvLine( I_in, 
					cvPoint( corners[i].x, corners[i].y-1 ), 
					cvPoint( corners[i].x, corners[i].y+1 ), 
					CV_RGB(255,0,0) );
		}
		cvSaveImage( OutFileName, I_in );
	}
	else if (inModeCrossesVsList == 2 ) 
	{
		// Print to txt file.
		printf( "Saving list of corners to file %s...", OutFileName );
		FILE* fp = fopen( OutFileName, "w" );
		for(int i=0; i < numCorners; i++ ) {
			fprintf( fp, "%i %i\n", corners[i].x, corners[i].y );
		}
		printf( "done.\n" );
		fclose( fp );
	}


}
