function [Xdesign,Mspline,spstruct,nprs] = mksplineSumDesignMat(X,ss)
% [Xdesign,Mspline,spstruct,nprs] = mksplineSumDesignMat(X,ss)
% 
% Create design matrix for a multivariate function parametrized as a
% sum-of-splines
%
% INPUTS: 
%          X - indep variables (each column is a regressor)
%         ss - cell array of structures with fields: 
%                "breaks", "smoothness", "extrapDeg"
%                (or use struct if only a single structure)
% 
% OUTPUTS: 
%          Xdesign - matrix relating 
%          Mspline - cell array of spline basis matrices (for low-d
%                         parametrization of spline)
%         spstruct - cell array of structures for each spline
%             nprs - vector with number of parameters for each spline.
%
% last updated: 7 Apr 2012 (JW Pillow)


nsplines = size(X,2);
nprs = zeros(nsplines,1);
Mspline = cell(1,nsplines);
spstruct = cell(1,nsplines);

% 1. Build design matrix for each spline
for ispl = 1:nsplines
   
    % Get spline structure for this column (regressor)
    if iscell(ss)
        spstruct{ispl} = ss{ispl};
    else
        spstruct{ispl} = ss;
    end
    ssi = spstruct{ispl};
    
    % Compute Design matrix for each spline
    if ispl == 1  
	% Keep DC for first spline
	[Xdesign,~,Mspline{ispl}] = mksplineDesignMat(X(:,ispl),[],ssi,0);
	nprs(ispl) = size(Mspline{ispl},2);
    else  
	% Remove DC of first segment for all subsequent splines
        [Xadd,~,Mspline{ispl}] = mksplineDesignMat(X(:,ispl),[],ssi,0,0);
	Xdesign = [Xdesign, Xadd];
	nprs(ispl) = size(Mspline{ispl},2);
    end
end
