function [XX,XY,khat] = fastLinRegress(X, Y, nkt, CriticalSize)
%  [XX, XY,khat] = fastLinRegress(X, Y, nkt, CriticalSize);
%
%  Computes the terms (X^T*X) and (X^T*Y) necessary to convolutionally regress
%  dependent variable Y against regressors X in a "causal" manner (i.e.,
%  Y(j) regressed against tim bins X(j-[0:nkt-1],:).
%
%  Note: standard regression solution for weights is K_ML = inv(XX)*XY; 
%
%  Input:  
%   X = (nT x nX) matrix of design variables (vertical dimension indicates time)
%   Y = (nT x 1) output variable (column vector)
%   nkt  = number of time samples of X to use to predict Y.
%   CriticalSize (optional) - max number of floats to store at once
%                  (smaller = slower but smaller memory requirement)
%
%  Output:
%   XY = X'*Y, projection of Y onto X  (vector of length nX*nkt).
%   XX = X'*X, stimulus covariance (nX*nKt) x (nX*nKt)
%   khat = ML regression solution XX\XY (optionally computed). 
%
%  Note: pads X with zeros for predicting earliest values of Y

%-------- Parse inputs  ---------------------------------------------------
if nargin < 4
    CriticalSize = 1e8; % max chunk size; decrease if getting "out of memory"
end
[nT,nX] = size(X);  % stimulus size (time bins x spatial bins).

rowlen = nX*nkt; % row length of design matrix
Msz = nT*rowlen; % # elements in full design matrix (if constructed)

% ------- Compute XX and XY -------------------------------------------
if Msz < CriticalSize  % Check if X is small enough to do in one chunk
    
    SS = makeStimRows(X,nkt);  % Make design matrix
    XY = SS'*Y;  % Compute desired quantities
    XX = SS'*SS;
        
else  % Compute Full X matrix in chunks, compute mean and cov on chunks
    nchunk = ceil(Msz/CriticalSize);
    chunksize = ceil(nT/nchunk);
    fprintf(1, 'fastLinRegress: using %d chunks\n', nchunk);
    
    % Compute on first chunk
    SS = makeStimRows(X(1:chunksize,:),nkt);  % make design matrix for first chunk
    Yvec = Y(1:chunksize);
    XY = SS'*Yvec;
    XX = SS'*SS;

    % Compute on remaining chunks
    nopadding = 1;
    for j = 2:nchunk;
        i0 = chunksize*(j-1)+1;  % starting index for chunk
        imax = min(nT,chunksize*j);  % ending index for chunk
        SS = makeStimRows(X(i0-nkt+1:imax,:),nkt,nopadding);
        Yvec = Y(i0:imax);
        
        XY = XY + SS'*Yvec;
        XX = XX + SS'*SS;
    end
end

% ------- Compute regression (ML) estimate for K ------------------------
if nargout > 2
    khat = XX\XY;
    khat = reshape(khat,nkt,nX);
end
