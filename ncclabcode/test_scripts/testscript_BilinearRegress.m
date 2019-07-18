% testscript_BilinearRegress.m 
%
% Tests out two variants of bilinear ridge regression with:
% (1) a bilinearly parametrized coefficient vector; 
% (2) a coefficient vector that is partly parametrized bilinearly, and the
% rest is linear. 

%% 1.  Bilinear least-squares regression 

nt = 25;
nx = 100;
p = 1; % rank

% Make filters
A = gsmooth(randn(nt,p),2);
B = gsmooth(randn(nx,p),2)';
nw = nt*nx;
wmat = A*B;
wtrue = vec(wmat);

subplot(211);
imagesc(wmat);

% Generate training data
nstim = 5000;
signse = 10;
X = randn(nstim,nw);
Y = X*wtrue + randn(nstim,1)*signse;
subplot(212);
plot(X*wtrue, Y, 'o');

%  Estimate W: coordinate ascent
XX = X'*X;
XY = X'*Y;
lambda = 1;

tic;
[what1,wt,wx] = bilinearRegress_coordAscent(XX,XY,[nt,nx],p,lambda);
toc;

% Estimate W: gradient-based ascent
tic;
[what2,wt2,wx2] = bilinearRegress_grad(XX,XY,[nt,nx],p,lambda);
toc;

% Plot filters and computer errors
subplot(211);
plot([wtrue what1(:) what2(:)]);
subplot(212);
imagesc([wmat, wt*wx, wt2*wx2]);
errs1 = [norm(wtrue-what1(:)), norm(wtrue-what2(:))]


%% 2. Part of the filter is bilinear, and part is linear
% -------------------------------------------------------

nt = 50;  % temporal length
nx = 20;  % spatial length
p = 2;    % rank
nwlin = 100;  % length of linear part

nwbi = nt*nx;  % number of filter elements in bilinear part
nwtot = nwbi+nwlin; % total number of filter elements

iibi = floor(nwlin/2)+(1:nwbi);  % indices for bilinear elements
iilin = setdiff(1:nwtot,iibi);   % indices for linear elements

% Make filters
A = gsmooth(randn(nt,p),2); % temporal filters
B = gsmooth(randn(nx,p),2)'; % spatial filters
wlin = gsmooth(randn(nwlin,1),10); % purely linear component

wbi = A*B;  % bilnear filter
wtrue = zeros(nwtot,1);  % composite filter
wtrue(iibi) = vec(wbi);
wtrue(iilin) = wlin;

% Generate training data
nstim = 5000;
signse = 10;
X = randn(nstim,nwtot);
Y = X*wtrue + randn(nstim,1)*signse;
XX = X'*X;
XY = X'*Y;

% make plots
subplot(221);
imagesc(wbi);
subplot(222);
plot(X*wtrue, Y, 'o');

% Estimate by Coordinate Ascent 
lambda = 10;  % ridge parameter
tic;
[what1,wt,wx,wlin] = bilinearMixRegress_coordAscent(XX,XY,[nt,nx],p,iibi,lambda);
toc;

% Estimate by Gradient Ascent
tic;
[what2,wt2,wx2,wlin2] = bilinearMixRegress_grad(XX,XY,[nt,nx],p,iibi,lambda);
toc;

% Assemble full weights for both cases

% Make plots
subplot(211);
plot([wtrue what1 what2]);
subplot(212);
imagesc([wbi, wt*wx, wt2*wx2]);
errs2 = [norm(wtrue-what1), norm(wtrue-what2)]
