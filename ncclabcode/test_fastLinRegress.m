% testRegression

% Make a test filter
nkt = 30;
k = 10*normpdf((1:nkt)'*[1 1 1], ones(nkt,1)*[10 15 20], 3);
plot(k);

% Convolve filter with white-noise stimulus
slen = 100000;
X = randn(slen,3);
Y = sameconv(X,k)+randn(slen,1)*1;

% Use fastCrossCov to estimate regression solution
tic;
[xx,xy] = fastCrossCov(X,Y,nkt);  % Use default block size (1e8)
toc;
khat1 = xx\xy;

% Plot results
subplot(211);
plot([k(:) khat1]);

%%

% Make a test filter
nkt = 30;
k = 10*normpdf((1:nkt)'*[1 1 1], ones(nkt,1)*[10 15 20], 3);
plot(k);

% Convolve filter with white-noise stimulus
slen = 100000;
X = randn(slen,3);
X(rand(slen,3)<.95) = 0;
Y = sameconv(X,k)+randn(slen,1)*1;

% Use fastCrossCov to estimate regression solution
tic;
[xx,xy] = fastCrossCov(X,Y,nkt);  % Use default block size (1e8)
toc;
khat1 = xx\xy;

% Use sparse version of X
Xsparse = sparse(X);
tic;
[xx,xy] = fastCrossCov(Xsparse,Y,nkt);  % Use default block size (1e8)
toc;