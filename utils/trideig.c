#include "mex.h"
#include <string.h>

/*   TRIDEIG Eigenvalues of symmetric tridiagonal matrix.   */

/*   P.-O. Persson <persson@mit.edu>   */
/*   Department of Mathematics, MIT    */
/*   October 21, 2002                  */

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
{
  double *D,*E;
  size_t n, info;

  if (nrhs!=2)
    mexErrMsgTxt("Must have exactly two input arguments.");
  if (nlhs>1)
    mexErrMsgTxt("Too many output arguments.");
  if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]))
    mexErrMsgTxt("Both inputs must be double arrays.");
  if (mxIsComplex(prhs[0]) || mxIsComplex(prhs[1]))
    mexErrMsgTxt("Both inputs must be real.");
  if (mxGetM(prhs[0])!=1 && mxGetN(prhs[0])!=1)
    mexErrMsgTxt("First input must be a vector.");
  if (mxGetM(prhs[1])!=1 && mxGetN(prhs[1])!=1)
    mexErrMsgTxt("Second input must be a vector.");

  n=mxGetNumberOfElements(prhs[0]);

  if (mxGetNumberOfElements(prhs[1])!=n-1)
    mexErrMsgTxt("The input vectors must have length N and N-1.");

  plhs[0]=mxDuplicateArray(prhs[0]);
  D=mxGetPr(plhs[0]);

  E=mxCalloc(n-1,sizeof(double));
  memcpy(E,mxGetPr(prhs[1]),(n-1)*sizeof(double));

  /* Remove _ on Windows */
  dsteqr_("N",&n,D,E,0,&n,0,&info);

  mxFree(E);

//   if (info)
//     mexErrMsgTxt("No convergence in DSTEQR.");
}
