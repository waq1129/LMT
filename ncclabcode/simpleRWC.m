function [RWA,RWC,RawMu,RawCov] = simpleRWC(Stim,resp,nkt,varargin)
% [RWA,RWC,RawMu,RawCov] = simpleRWC(Stim,resp,nkt,MaxSize)
%
% Computes response-weighted mean and covariance, which are analogous to
% spike-triggered average and covariance. 
%
% RWA differs from STA only because it is not normalized by the total
% spike count.
%
% RWC differs from STC because it is not normalized by spike count and it
% is not centered around the RWA (i.e., is zero-centered)
%
% INPUT:
%    Stim [N x M]   - stimulus matrix; 1st dimension = time, 2nd dimension = space
%    resp [N x 1]   - column vector of spike count in each time bin (can be sparse)
%     nkt [1 x 1]   - # time samples to consider to be part of the stimulus
% MaxSize [1 x 1] (optional)  - max # of floats to store while computing cov
%                              (smaller = slower, but smaller memory requirement)
%  OUTPUT:
%    RWA [nkt x M]       - spike-triggered average (reshaped as matrix)
%    RWC [nkt*M x nkt*M] - spike-triggered covariance (covariance around the mean);
%  RawMu [nkt*M x 1]     - mean of raw stimulus ensemble
% RawCov [nkt*M x nkt*M] - covariance of raw stimulus ensemble
%
%  Notes:  
%  (1) Ignores response before "nkt" time bins
%  (2) Faster if only 2 output arguments (raw mean and covariance not computed)
%  (3) Reduce 'maxsize' if getting "out of memory" errors
%  (4) If resp is non-integer, computes response-weighted mean and
%  covariance, meaning moments are mul divided by sum(resp) and (sum(resp)-1), respectively, meaning one
%  probably wants to multiply by sum(resp) and sum(resp)-1 to get the correct
%  "response weighted mean" and "response weighted covariance".
%
%  --------
%  Details:
%  --------
%   Let X = "valid" design matrix (from makeStimRows)
%       Y = resp (spike train)
%     nsp = sum(Y); % number of spikes (sum of responses)
%       N = size(Stim,1); % number of stimuli
%
%   then  RWA = X'*Y 
%         RWA = X'*(X.*repmat(Y,1,ncols))
%         RawMu = sum(X)/N;
%         RawCov = X*X'/(N-1);
%
% $Id$


resp(1:nkt-1) = 0; % Remove response in first nkt-1 bins
totresp = sum(resp); % Compute sum of responses
if totresp==0 % Guard against case where responses have mean zero (matched in simpleSTC.m)
    totresp = 2;
end

if nargout <= 2
    % Compute RWA and RWC only
    [sta,stc] = simpleSTC(Stim,resp,nkt,varargin{:});
    RWA = sta*totresp;
    RWC = (stc + sta(:)*sta(:)'*totresp/(totresp-1))*(totresp-1);
else
    % Compute RWA and RWC and raw mean and covariance
    [sta,stc,RawMu,RawCov] = simpleSTC(Stim,resp,nkt,varargin{:});
    RWA = sta*totresp;
    RWC = (stc + sta(:)*sta(:)'*totresp/(totresp-1))*(totresp-1);
end
