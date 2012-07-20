
#include <math.h>
#include <fstream>
using std::ifstream;

//typedef unsigned short CHAR16_T;
#include "mex.h"
#include "matrix.h"

static bool Debug = false;

double getDoubleField(const mxArray* options, const char* name) {
	return mxGetScalar(mxGetField(options, 0, name));
}

bool getBooleanField(const mxArray* options, const char* name) {
	return mxIsLogicalScalarTrue(mxGetField(options, 0, name));
}

double getPrediction(double* U, double* V, int i, int j, int numNodes, int dim, bool symm) {
	double res = 0;
	for (int k=0; k<dim; ++k) {
		res += U[i+k*numNodes] * V[j+k*numNodes];
		if (symm)
			res += V[i+k*numNodes] * U[j + k*numNodes];
	}
	return res;
}

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

double sum_squares(double* U, int i, int numNodes, int dim) {
	double res = 0;
	for(int k = 0; k < dim; ++k)
		res += U[i+k*numNodes] * U[i+k*numNodes];
	return res;
}

double sum_abs(double* U, int i, int numNodes, int dim) {
	double res = 0;
	for(int k = 0; k < dim; ++k)
		res += abs(U[i+k*numNodes]);
	return res;
}

void iterate(double* M, double* neighbor_list, double* U0, double* V0, int numNodes, int numSamples, int dim, const mxArray* options, mxArray* plhs[]) {	
	double learn_rate = getDoubleField(options, "learn_rate");
	double lambda = getDoubleField(options, "huber_lambda");
	double L2Reg = getDoubleField(options, "L2Reg");
	double L1Reg = getDoubleField(options, "L1Reg");
	bool symm = getBooleanField(options, "symm_factorization");
	if (Debug) {
		mexPrintf("numNodes = %d, numSamples = %d\n", numNodes, numSamples);
		mexPrintf("L1Reg = %.3f, L2Reg = %.3f\n", L1Reg, L2Reg);
		
	}
	
	plhs[2] = mxCreateDoubleMatrix(numNodes, dim, mxREAL);
	plhs[3] = mxCreateDoubleMatrix(numNodes, dim, mxREAL);
	double* U = mxGetPr(plhs[2]);
	double* V = mxGetPr(plhs[3]);
	for (int i=0; i<numNodes; ++i) {
		for (int j=0; j<dim; ++j) {
			U[i+j*numNodes] = U0[i+j*numNodes];
			V[i+j*numNodes] = V0[i+j*numNodes];
		}
	}
	
	int* numNeighborsU = new int[numNodes];
	int* numNeighborsV = new int[numNodes];

	for (int i=0; i<numNodes; ++i) {
		numNeighborsU[i] = numNeighborsV[i] = 0;
	}
	
	for(int i=0; i<numSamples; ++i) {
		int n1 = (int)neighbor_list[i] - 1;
		int n2 = (int)neighbor_list[i+numSamples] - 1;
		numNeighborsU[n1]++;
		numNeighborsV[n2]++;
	}
	
	int dim2 = 2;
	for(int i=0; i<numSamples; ++i) {
		int n1 = (int)neighbor_list[i] - 1;
		int n2 = (int)neighbor_list[i+numSamples] - 1;
		double latency = M[n1+n2*numNodes];
		double pred = getPrediction(U, V, n1, n2, numNodes, dim, symm);
		double f, g;
		huber(pred - latency, lambda, f, g);
		
		for(int k=0; k<dim; ++k) {
			double& u_f = U[n1+k*numNodes];
			double& i_f = V[n2+k*numNodes];
			double delta_u = - g * i_f - (L1Reg * sign(u_f) + 2 * L2Reg * u_f);
			double delta_i = - g * u_f - (L1Reg * sign(i_f) + 2 * L2Reg * i_f);
			u_f += delta_u * learn_rate;
			i_f += delta_i * learn_rate;
		}
		
		if (symm) {
			for(int k=0; k<dim; ++k) {
				double& u_f = U[n2+k*numNodes];
				double& i_f = V[n1+k*numNodes];
				double delta_u = - g * i_f - (L1Reg * sign(u_f) + 2 * L2Reg * u_f);
				double delta_i = - g * u_f - (L1Reg * sign(i_f) + 2 * L2Reg * i_f);
				u_f += delta_u * learn_rate;
				i_f += delta_i * learn_rate;
			}

		}
	}
	
	double loss = 0, complexity = 0;
	for (int i=0; i<numSamples; ++i) {
		int n1 = (int)neighbor_list[i] - 1;
		int n2 = (int)neighbor_list[i+numSamples] - 1;
		double latency = M[n1+n2*numNodes];
		double pred = getPrediction(U, V, n1, n2, numNodes, dim, symm);
		double f, g;
		huber(pred - latency, lambda, f, g);
		
		loss += f;
	}
	
	int ratio = symm ? 2 : 1;
	for (int i=0; i<numNodes; ++i) {
		complexity += L2Reg * sum_squares(U, i, numNodes, dim) * numNeighborsU[i] * ratio;
		complexity += L1Reg * sum_abs(U, i, numNodes, dim) * numNeighborsU[i] * ratio;
		complexity += L2Reg * sum_squares(V, i, numNodes, dim) * numNeighborsV[i] * ratio;
		complexity += L1Reg * sum_abs(V, i, numNodes, dim) * numNeighborsV[i] * ratio;
	}
	
	plhs[0] = mxCreateDoubleScalar(loss);
	plhs[1] = mxCreateDoubleScalar(complexity);
	
	delete[] numNeighborsU;
	delete[] numNeighborsV;
}

void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]) {

	if (nrhs!=5) {
		mexErrMsgTxt("Five inputs (M, W, U, V, options) required.");
	}
	
	if (nlhs != 4) {
		mexErrMsgTxt("Four outputs required: loss, complexity, U, V");
	}
	double* M = mxGetPr(prhs[0]);
	double* neighbor_list = mxGetPr(prhs[1]);
	double* U = mxGetPr(prhs[2]);
	double* V = mxGetPr(prhs[3]);
	const mxArray* options = prhs[4];
	Debug = getDoubleField(options, "Debug") == 1;
	
	
	int numNodes = (int)mxGetM(prhs[0]);
	int numSamples = (int)mxGetM(prhs[1]);
	int dim = (int)mxGetN(prhs[2]);
	mxAssert(mxGetN(prhs[1]) == 2, "2nd dimension of neighbor_list should be 2");
	mxAssert(mxGetN(prhs[0]) == numNodes, "M should be square matrix");
	
	
	iterate(M, neighbor_list, U, V, numNodes, numSamples, dim, options, plhs);
}
