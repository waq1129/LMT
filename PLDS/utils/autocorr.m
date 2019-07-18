function varargout = autocorr(y,numLags,numMA,numSTD)
%AUTOCORR Sample autocorrelation
%
% Syntax:
%
%   [acf,lags,bounds] = autocorr(y)
%   [acf,lags,bounds] = autocorr(y,numLags,numMA,numSTD)
%   autocorr(...)
%
% Description:
%
%   Compute the sample autocorrelation function (ACF) of a univariate, 
%   stochastic time series y. When called with no output arguments,
%   AUTOCORR plots the ACF sequence with confidence bounds.
%
% Input Arguments:
%
%   y - Vector of observations of a univariate time series for which the
%     sample ACF is computed or plotted. The last element of y contains the
%     most recent observation.
%
% Optional Input Arguments:
%
%   numLags - Positive integer indicating the number of lags of the ACF 
%     to compute. If empty or missing, the default is to compute the ACF at 
%     lags 0,1,2, ... T = min[20,length(y)-1]. Since ACF is symmetric
%     about lag zero, negative lags are ignored.
%
%   numMA - Nonnegative integer indicating the number of lags beyond which 
%     the theoretical ACF is deemed to have died out. Under the hypothesis
%     that the underlying y is really an MA(numMA) process, the large-lag
%     standard error is computed via Bartlett's approximation for lags >
%     numMA as an indication of whether the ACF is effectively zero beyond
%     lag numMA. If numMA is empty or missing, the default is numMA = 0, in
%     which case y is assumed to be Gaussian white noise. If y is a
%     Gaussian white noise process of length N, the standard error will be
%     approximately 1/sqrt(N). numMA must be less than numLags.
%
%   numSTD - Positive scalar indicating the number of standard deviations
%     of the sample ACF estimation error to compute, assuming the
%     theoretical ACF of y is zero beyond lag numMA. When numMA = 0 and y
%     is a Gaussian white noise process of length numMA, specifying numSTD
%     will result in confidence bounds at +/-(numSTD/sqrt(numMA)). If empty
%     or missing, the default is numSTD = 2 (approximate 95% confidence).
%
% Output Arguments:
%
%   acf - Sample autocorrelation function of y. acf is a vector of 
%     length numLags+1 corresponding to lags 0,1,2,...,numLags. The first 
%     element of acf is unity (i.e., acf(1) = 1 at lag 0).
%
%   lags - Vector of lags corresponding to acf (0,1,2,...,numLags).
%
%   bounds - Two-element vector indicating the approximate upper and lower
%     confidence bounds, assuming that y is an MA(numMA) process. Note that 
%     bounds is approximate for lags > numMA only.
%
% Example:
%
%   % Create an MA(2) process from a sequence of 1000 Gaussian deviates,
%   % and assess whether the ACF is effectively zero for lags > 2:
%
%     x = randn(1000,1);         % 1000 Gaussian deviates ~ N(0,1)
%     y = filter([1 -1 1],1,x);  % Create an MA(2) process
%     autocorr(y,[],2)           % Inspect the ACF with 95% confidence
%
% Reference:
%
%   [1] Box, G. E. P., G. M. Jenkins, and G. C. Reinsel. Time Series
%       Analysis: Forecasting and Control. 3rd edition. Upper Saddle River,
%       NJ: Prentice-Hall, 1994.
%
% See also CROSSCORR, PARCORR, FILTER.

% Copyright 1999-2010 The MathWorks, Inc.   
% $Revision: 1.1.8.4 $  $Date: 2010/10/08 16:41:02 $

% Ensure the sample data is a vector:

[rows,columns] = size(y);

if (rows ~= 1) && (columns ~= 1)
    
    error(message('econ:autocorr:NonVectorInput'))
      
end

rowSeries = (size(y,1) == 1);

y = y(:);         % Ensure a column vector
N = length(y);    % Sample size
defaultLags = 20; % Recommendation of [1]

% Ensure numLags is a positive integer or set default:

if (nargin >= 2) && ~isempty(numLags)
    
   if numel(numLags) > 1
       
      error(message('econ:autocorr:NonScalarLags'))
        
   end
   
   if (round(numLags) ~= numLags) || (numLags <= 0)
       
      error(message('econ:autocorr:NonPositiveInteger'))
        
   end
   
   if numLags > (N-1)
       
      error(message('econ:autocorr:LagsTooLarge'))
        
   end
   
else
    
   numLags = min(defaultLags,N-1); % Default
   
end


% Ensure numMA is a nonnegative integer or set default:

if (nargin >= 3) && ~isempty(numMA)
    
   if numel(numMA) > 1
       
      error(message('econ:autocorr:NonScalarNMA'))
        
   end
   
   if (round(numMA) ~= numMA) || (numMA < 0)
       
      error(message('econ:autocorr:NegativeIntegerNMA'))
        
   end
   
   if numMA >= numLags
       
      error(message('econ:autocorr:NMATooLarge'))
        
   end
   
else
    
   numMA = 0; % Default
   
end

% Ensure numSTD is a positive scalar or set default:

if (nargin >= 4) && ~isempty(numSTD)
    
   if numel(numSTD) > 1
       
      error(message('econ:autocorr:NonScalarSTDs'))
        
   end
   
   if numSTD < 0
       
      error(message('econ:autocorr:NegativeSTDs'))
        
   end
   
else
    
   numSTD = 2; % Default
   
end

% Convolution, polynomial multiplication, and FIR digital filtering are all
% the same operation. The FILTER command could be used to compute the ACF
% (by convolving the de-meaned y with a flipped version of itself), but
% FFT-based computation is significantly faster for large data sets.

% The ACF computation is based on [1], pages 30-34, 188:

nFFT = 2^(nextpow2(length(y))+1);
F = fft(y-mean(y),nFFT);
F = F.*conj(F);
acf = ifft(F);
acf = acf(1:(numLags+1)); % Retain non-negative lags
acf = acf./acf(1); % Normalize
acf = real(acf);

% Compute approximate confidence bounds using the approach in [1],
% equations 2.1.13 and 6.2.2, pp. 33 and 188, respectively:

sigmaNMA = sqrt((1+2*(acf(2:numMA+1)'*acf(2:numMA+1)))/N);  
bounds = sigmaNMA*[numSTD;-numSTD];
lags = (0:numLags)';

if nargout == 0

%  Plot the sample ACF:

   lineHandles = stem(lags,acf,'filled','r-o');
   set(lineHandles(1),'MarkerSize',4)
   grid('on')
   xlabel('Lag')
   ylabel('Sample Autocorrelation')
   title('Sample Autocorrelation Function')
   hold('on')

%  Plot confidence bounds (horizontal lines) under the hypothesis that the
%  underlying y is really an MA(numMA) process. Bartlett's approximation
%  gives an indication of whether the ACF is effectively zero beyond lag
%  numMA. For this reason, the confidence bounds appear over the ACF only
%  for lags greater than numMA (i.e., numMA+1, numMA+2, ... numLags). In
%  other words, the confidence bounds enclose only those lags for which the
%  null hypothesis is assumed to hold. 

   plot([numMA+0.5 numMA+0.5; numLags numLags],[bounds([1 1]) bounds([2 2])],'-b');
   plot([0 numLags],[0 0],'-k');
   hold('off')
   a = axis;
   axis([a(1:3) 1]);

else

%  Re-format outputs for compatibility with the y input. When y is input as
%  a row vector, then pass the outputs as a row vectors; when y is a column
%  vector, then pass the outputs as a column vectors.

   if rowSeries
       
      acf = acf';
      lags = lags';
      bounds = bounds';
      
   end

   varargout = {acf,lags,bounds};

end