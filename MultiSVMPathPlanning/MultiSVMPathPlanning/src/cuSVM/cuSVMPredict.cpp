#include <mat.h>
#include <mex.h>


extern "C"
void GPUPredictWrapper (int m, int n, int k, float kernelwidth, const float *Test, const float *Svs, float * alphas,float *prediction, float beta,float isregression);

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )

{
	
	int m=mxGetM(prhs[0]);
	int k=mxGetN(prhs[0]);
	
	//B is transposed in the RBFKernel gpu function
	int n=mxGetM(prhs[1]);
	int testk=mxGetN(prhs[1]);
	
	if (m>60000)
		mexErrMsgTxt("cuSVMPredict's maximum number of test points is 60,000.");

	if (nrhs!=6)
		mexErrMsgTxt("cuSVMPredict must have six, and only six, inputs.");

	if (nlhs>1)
		mexErrMsgTxt("cuSVMPredict only has one output");

	if(testk!=k)
		mexErrMsgTxt("The test data and the support vectors must have the same number of features (columns).");

	if(n!=mxGetM(prhs[2]))
		mexErrMsgTxt("The number of support vectors must be equal to the number of alpha coefficients.");


	if (mxIsClass(prhs[0], "single") + mxIsClass(prhs[1], "single")+mxIsClass(prhs[2], "single")+mxIsClass(prhs[3], "single")!=4)
		mexErrMsgTxt("The test data matrix, support vector matrix, alpha weight vector, and beta scalar all must be single precision floats.");



	float* Test=(float *)mxGetData(prhs[0]);
	float* Svs=(float *)mxGetData(prhs[1]);
	float* alphas=(float *)mxGetData(prhs[2]);
	float beta=*(float *)mxGetData(prhs[3]);

	float kernelwidth;
	float isregression;

	if (mxIsClass(prhs[4],"double")==1)
		kernelwidth=(float)*(double*)mxGetData(prhs[4]);
	else if (mxIsClass(prhs[4],"single")==1)
		kernelwidth=*(float*)mxGetData(prhs[4]);
	else
		mexErrMsgTxt("The kernel width and regression indicator variables both must be either single or double precision floats.");
	
	if (mxIsClass(prhs[5],"double")==1)
		isregression=(float)*(double*)mxGetData(prhs[5]);
	else if (mxIsClass(prhs[5],"single")==1)
		isregression=*(float*)mxGetData(prhs[5]);
	else
		mexErrMsgTxt("The kernel width and regression indicator variables both must be either single or double precision floats.");

	if(isregression!=1 && isregression!=0)
		mexErrMsgTxt("The regression indicator variable must be either zero or one.");
	
	
	plhs[0]=mxCreateNumericMatrix(m,1,mxSINGLE_CLASS, mxREAL);
	
	float* prediction=(float *)mxGetData(plhs[0]);


	GPUPredictWrapper(m,n,k,kernelwidth,Test,Svs,alphas,prediction,beta,isregression);

	


	return;
}
