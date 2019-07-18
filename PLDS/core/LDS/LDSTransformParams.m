function [params, seq] = LDSTransformParams(params,varargin)
%
% function [params, seq] = LDSTransformParams(params,varargin)
%
%
% transform parameters of LDS by imposing constraints on C and the
% stationary distribution Pi.
%
%  Pi := dlyap(params.model.A,params.model.Q)   stationary distribution
%
% TransformType:
%
% 0: do nothing
% 1: C'*C = eye		&&		Pi = diag    [default]
% 2: C'*C = diag 	&& 		Pi = eye
% 3: ||C(:,k)||_2 = 1
% 4: P_ii = 1
% 5: A = blk-diag       &&              ||C(:,k)||_2 = 1
%
% (c) 2014 Lars Busing   lars@stat.columbia.edu
%
%
% also see: LDSApplyParamsTransformation(M,params,varargin)
%


seq = [];
TransformType   = '1';

assignopts(who,varargin);

xDim = size(params.model.A,1);

switch TransformType

 case '0'
  
  % do nothing
  
 case '1'
  
  [UC,SC,VC]    = svd(params.model.C,0);
  [params seq]  = LDSApplyParamsTransformation(SC*VC',params,'seq',seq);
  
  params.model.Pi     = dlyap(params.model.A,params.model.Q);
  if min(eig(params.model.Pi))<0
    params.model.Pi = params.model.Q;
  end
  [UPi SPi VPi] = svd(params.model.Pi);
  [params seq]  = LDSApplyParamsTransformation(UPi',params,'seq',seq);
  
 case '2'  
  
  params.model.Pi = dlyap(params.model.A,params.model.Q);
  if min(eig(params.model.Pi))<0
    params.model.Pi = params.model.Q;
  end
  [UPi SPi VPi] = svd(params.model.Pi);
  M    	      = diag(1./sqrt(diag(SPi)))*UPi';
  [params seq]  = LDSApplyParamsTransformation(M,params,'seq',seq);    	
  
  [UC,SC,VC]    = svd(params.model.C,0);
  [params seq]  = LDSApplyParamsTransformation(VC',params,'seq',seq);  	
  
 case '3'

  [params seq] = LDSApplyParamsTransformation(diag(sqrt(sum(params.model.C.^2,1))),params,'seq',seq);

 case '4'

  Pi = dlyap(params.model.A,params.model.Q);
  if min(eig(Pi))<0
    Pi = params.model.Q;
  end
  M = diag(1./sqrt(diag(Pi)));
  [params seq] = LDSApplyParamsTransformation(M,params,'seq',seq);
  
 case '5'

  [T B] = bdschur(params.model.A,inf);
  [params seq] = LDSApplyParamsTransformation(pinv(T),params);
  [params seq] = LDSApplyParamsTransformation(diag(sqrt(sum(params.model.C.^2,1))),params,'seq',seq);


 otherwise
  
  warning('Unknow paramter transformation type')
  
end
