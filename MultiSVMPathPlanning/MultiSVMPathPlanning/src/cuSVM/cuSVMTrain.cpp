#include <mat.h>
#include <mex.h>

extern "C"
void SVMTrain(float *mexalpha,float* beta,float*y,float *x ,float C, float kernelwidth, int m, int n, float StoppingCrit);

extern "C"
void SVRTrain(float *mexalpha,float* beta,float*y,float *x ,float C, float kernelwidth, float eps, int m, int n, float StoppingCrit);

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )

{
	if (nlhs>3)
		mexErrMsgTxt("cuSVMTrain has at most 3 outputs.");
	
	if (nrhs>6)
		mexErrMsgTxt("Too many input arguments.");

	if (nrhs<5)
		mexErrMsgTxt("Too few input arguments.");


	if (mxIsClass(prhs[0], "single") + mxIsClass(prhs[1], "single")!=2)
		mexErrMsgTxt("Both the target vector and feature matrix must consist of single precision floats.");
	
	int n=mxGetN(prhs[1]);
	int m=mxGetM(prhs[1]);
	
	if (mxGetM(prhs[0])!=m)
		mexErrMsgTxt("The target vector and feature matrix must have the same number of rows.");

	if (mxGetN(prhs[0])!=1)
		mexErrMsgTxt("The target vector must only have one column.");

	if ((mxGetM(prhs[2])!=1) | (mxGetN(prhs[2])!=1)|(mxGetM(prhs[3])!=1) | (mxGetN(prhs[3])!=1)|((nrhs>=5&&(mxIsEmpty(prhs[4])!=1))?(mxGetM(prhs[4])!=1) | (mxGetN(prhs[4])!=1):0)|(nrhs==6?(mxGetM(prhs[5])!=1) | (mxGetN(prhs[5])!=1):0))
		mexErrMsgTxt("The regularization parameter (C), the kernel width, epsilon, and the stopping criterion (if specified) all must be scalars.");

	



	float C;
	float kernelwidth;
	float eps;
	

	if (mxIsClass(prhs[2],"double")==1)
		C=(float)*(double *)mxGetData(prhs[2]);
	else if (mxIsClass(prhs[2],"single")==1)
		C=*(float *)mxGetData(prhs[2]);
	else
		mexErrMsgTxt("The regularization parameter (C), the kernel width, epsilon, and the stopping criterion (if specified) all must be either single or double precision floats.");
	
	if (mxIsClass(prhs[3],"double")==1)
		kernelwidth=(float)*(double *)mxGetData(prhs[3]);
	else if (mxIsClass(prhs[3],"single")==1)
		kernelwidth=*(float *)mxGetData(prhs[3]);
	else
		mexErrMsgTxt("The regularization parameter (C), the kernel width, epsilon, and the stopping criterion (if specified) all must be either single or double precision floats.");
	if (kernelwidth<=0)
		mexErrMsgTxt("The kernel width must be greater than zero.");
	

	int IsRegression=0;

	if (mxIsEmpty(prhs[4])!=1)
	{	
		IsRegression=1;
		
		if (mxIsClass(prhs[4],"double")==1)
			eps=(float)*(double *)mxGetData(prhs[4]);
		else if (mxIsClass(prhs[4],"single")==1)
			eps=*(float *)mxGetData(prhs[4]);
		else
			mexErrMsgTxt("The regularization parameter (C), the kernel width, epsilon, and the stopping criterion (if specified) all must be either single or double precision floats.");
	}
	
	float StoppingCrit=0.001;
	
	if (nrhs==6)
	{

		if (mxIsClass(prhs[5],"double")==1)
			StoppingCrit=(float)*(double *)mxGetData(prhs[5]);
		else if (mxIsClass(prhs[5],"single")==1)
			StoppingCrit=*(float *)mxGetData(prhs[5]);
		else
			mexErrMsgTxt("The regularization parameter (C), the kernel width, epsilon, and the stopping criterion (if specified) all must be either single or double precision floats.");

		if ((StoppingCrit<=0) | (StoppingCrit>=.5) )
			mexErrMsgTxt("The stopping criterion must be greater than zero and less than .5.");

	}	


	float* y=(float *)mxGetData(prhs[0]);
	float* x=(float *)mxGetData(prhs[1]);

	plhs[1]=mxCreateNumericMatrix(1, 1,mxSINGLE_CLASS, mxREAL);
	float *beta=(float*)mxGetData(plhs[1]);

	float *alpha=new float [m];

	if (IsRegression)
	{

		SVRTrain(alpha,beta,y,x ,C,kernelwidth,eps ,m,n,StoppingCrit);
	}
	else
	{
		int JustOneClassError=1;
		int NotOneorNegOneError=0;
		float FirstY=y[0];

		for(int k=0;k<m;k++)
		{
			if(y[k]!=FirstY) {JustOneClassError=0;}
			if((y[k]!=1.0) && (y[k]!=-1.0) ){NotOneorNegOneError=1;}
		}
		
		if (JustOneClassError==1)
			mexErrMsgTxt("All training labels are of the same class.  There must of course be two classes");
		
		if (NotOneorNegOneError==1)
			mexErrMsgTxt("Training labels must be either 1 or -1.");

		SVMTrain(alpha,beta,y,x ,C,kernelwidth,m,n,StoppingCrit);
	}
	
	
	int numSVs=0;
	int numPosSVs=0;
	for(int k=0;k<m;k++)
	{
		if(alpha[k]!=0)
		{
			if(IsRegression==0) 
			{
				alpha[k]*=y[k];
				if(y[k]>0) {numPosSVs++;}
			}
			
			numSVs++;
		}
	}
	
	
	plhs[0]=mxCreateNumericMatrix(numSVs, 1,mxSINGLE_CLASS, mxREAL);
	float *SvAlphas=(float*)mxGetData(plhs[0]);
	
	plhs[2]=mxCreateNumericMatrix(numSVs, n,mxSINGLE_CLASS, mxREAL);
	float *Svs=(float*)mxGetData(plhs[2]);

	

	if(IsRegression==0) 
	{
	
		int PosSvIndex=0;
		int NegSvIndex=0;

		for(int k=0;k<m;k++)
		{
			if(alpha[k]!=0)
			{
				if(y[k]>0)
				{
					SvAlphas[PosSvIndex]=alpha[k];
					for(int j=0;j<n;j++)
					{Svs[PosSvIndex+j*numSVs]=x[k+j*m];}
					PosSvIndex++;
				}
				else
				{
					SvAlphas[NegSvIndex+numPosSVs]=alpha[k];
					for(int j=0;j<n;j++)
					{Svs[NegSvIndex+numPosSVs+j*numSVs]=x[k+j*m];}
					NegSvIndex++;
				}			
			}		
		}
	}
	else
	{
		int svindex=0;

		for(int k=0;k<m;k++)
		{
			if(alpha[k]!=0)
			{
				SvAlphas[svindex]=alpha[k];
				for(int j=0;j<n;j++)
				{Svs[svindex+j*numSVs]=x[k+j*m];}
				svindex++;
			}
		
		}

	 }



	return;
}
