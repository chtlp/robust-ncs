
#include "math.h"
#include <fstream>
#include <algorithm>
#include <assert.h>
using std::ifstream;
using std::random_shuffle;
using std::copy;

//typedef unsigned short CHAR16_T;
#ifdef _CHAR16T
#define CHAR16_T
#endif

#include "mex.h"
#include "matrix.h"

static bool Debug = false;



void huber(double x, double lambda, double& f, double& g) {
	if (abs(x) < lambda) {
		f = x * x / 2 / lambda;
		g = x / lambda;
	}
	else {
		f = abs(x) - lambda / 2;
		g = x > 0 ? 1 : -1;
	}
}

double sign(double x) {
	return x > 0 ? 1 : (x < 0 ? -1 : 0);
}

double get_prediction(double* C, int n, int dim, int u, int v){
	double sq = 0;
	for(int i=0; i<dim; ++i) {
		double e = C[u+i*n] - C[v+i*n];
		sq += e * e;
	}
	return sqrt(sq) + abs(C[u+dim*n]) + abs(C[v+dim*n]);
}

double iterate(double* M, double* C, double* sample_list, int n, int dim, int num_samples, double learn_rate, double lambda) {
	for(int i=0; i<num_samples; ++i) {
		int u = (int) sample_list[i] - 1, v = (int) sample_list[i+num_samples] - 1;
		if (u == v) continue;

		//assert(u >= 0 && u < n && v >= 0 && v < n);
		
		double measure = M[u+n*v];
		double predict = get_prediction(C, n, dim, u, v);
		double f, g;
		huber(predict - measure, lambda, f, g);
		
		double grad_common = g / predict;
		
		for(int k=0; k<dim; ++k) {
			double x1 = C[u+k*n], x2 = C[v+k*n];
			C[u+k*n] -= learn_rate * grad_common * (x1-x2);
			C[v+k*n] -= learn_rate * grad_common * (x2-x1);

			//if (C[u+k*n] != C[u+k*n] || C[v+k*n] != C[v+k*n]) {
			//	mexPrintf("worng\n");
			//}
		}
		
		C[u+dim*n] -= learn_rate * g * sign(C[u+dim*n]);
		C[v+dim*n] -= learn_rate * g * sign(C[v+dim*n]);
		
		if (C[u+dim*n] < 0)
			C[u+dim*n] = 0;
		
		if (C[v+dim*n] < 0)
			C[v+dim*n] = 0;
	}
	
	double loss = 0;
	for(int i=0; i<num_samples; ++i) {
		int u = (int) sample_list[i] - 1, v = (int) sample_list[i+num_samples] - 1;
		if (u == v) continue;

		double measure = M[u+n*v];
		double predict = get_prediction(C, n, dim, u, v);
		double f, g;
		huber(measure - predict, lambda, f, g);
		
		loss += f;
	}
	
	//mexPrintf("loss = %.3f\n", loss);
	return loss;
}

void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]) {
	

	
	if (nrhs!=5) {
		mexErrMsgTxt("Four inputs (M, sample_list, C0, learn_rate, lambda) required.");
	}
	
	if (nlhs != 2) {
		mexErrMsgTxt("Two outputs required: C, loss");
	}
	
	double* M = mxGetPr(prhs[0]);
	double* sample_list = mxGetPr(prhs[1]);
	double* C0 = mxGetPr(prhs[2]);	
	double learn_rate = mxGetScalar(prhs[3]);
	double lambda = mxGetScalar(prhs[4]);
	
	int m = (int) mxGetM(prhs[0]);
	int n = (int) mxGetN(prhs[0]);
	int num_samples = (int) mxGetM(prhs[1]);
		
	int dim = (int) mxGetN(prhs[2]) - 1;
	
	if (Debug)
		mexPrintf("m = %d, n = %d, num_samples = %d, dim = %d\n", m, n, num_samples, dim);
		

	plhs[0] = mxCreateDoubleMatrix(n, dim+1, mxREAL);
	
	double* C = mxGetPr(plhs[0]);
	
	copy(C0, C0+n*(dim+1), C);
	
	double loss = iterate(M, C, sample_list, n, dim, num_samples, learn_rate, lambda);
	
	plhs[1] = mxCreateDoubleScalar(loss);
}
