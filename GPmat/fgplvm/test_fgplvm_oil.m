% simple script to test brain kernel embedding idea: FITTING
clear,clf,clc,addpath(genpath(pwd)); warning off

dataSetName = 'oil';
experimentNo = 1;
% load data
[Y0, lbls] = lvmLoadData(dataSetName);
[nx_true, nf0] = size(Y0);

% squared exponential (SE) covariance function
% kSE = @(r,l,x)(r*exp(-.5*(bsxfun(@minus,x(:),x(:)')/l).^2));
kSE = @(l,r,x) covSEiso([log(l);log(r)], x) ;
klinear = @(r,l,x) x*x';
matrix_mse = @(x) sqrt(sum(vec(x.^2)));

% Set up grid for visualization
nc = length(nx_true);
gridends = [ones(nc,1) nx_true(:)]; % range of function to consider
varargin = cell(nc,1);
varargout = cell(nc,1);
for ii=1:nc
    varargin{ii} = gridends(ii,1):gridends(ii,2);
end
[varargout{1:nc}] = ndgrid(varargin{:});
xgrid = []; for ii=1:nc xgrid = [xgrid vec(varargout{ii})]; end

% Generate some y samples
ysamps = Y0';
[nsamp,nx] = size(ysamps);
ysamps = ysamps-repmat(mean(ysamps,1),nsamp,1);
subplot(221);
plot(ysamps(1:10,:)');

% Compute stuff needed for fits
Ycov = ysamps'*ysamps;
Cmat = Ycov/nsamp;
subplot(222);
imagesc(Ycov),axis image,colorbar

%% initialize
nf = 2;
opts = optimset('display', 'iter', 'largescale', 'off','maxiter',1e6,'maxfunevals',1e6);

Y = ysamps';
options = fgplvmOptions('ftc');
d = size(Y, 2);
model = fgplvmCreate(nf, d, Y, options);
params0 = fgplvmExtractParam(model);
alpha0 = rand(nc,nf)*0; f0 = params0(1:nx*nf)'; prs0 = [alpha0(:); f0];
alpha = alpha0;

%%
figure(2)
iters = 100;
display = 1;
mflag = 1;
if mflag
    options = [];
    options.Method='lbfgs';
    options.TolFun=1e-10;
    options.MaxIter = 2e3;
    options.maxFunEvals = 2e3;
    options.Display = 'iter';
else
    options = optimset('GradObj','off','display', 'iter', 'largescale', 'off','maxiter',1e6,'maxfunevals',1e6);
end

%% test anqiwu with rbf kernel with gp prior, optimize over 3 hyparams, no biasnfln = round(nx/3);
figure(2)
S = Ycov/nsamp;
[vv,ee] = svd(S);
de = diag(ee);
cde = cumsum(de);
cc = cde<cde(end)*0.95;
[aa,bb] = find(cc==1);
nfln = aa(end)+1;
ll = diag(ee(nfln:end,nfln:end));
nsevar_est = sum(ll)/length(ll);
rho_est = mean(diag(S))-nsevar_est;
len_est = 10;
hyp = [-2*log(len_est);log(rho_est);log(nsevar_est)];
palpha = alpha(:);
lambda = 0;
KpriorInv = eye(nx);
Cnse = zeros(nx);
fmu = xgrid*alpha;

lfunc = @(pp) compNegEmbeddedGPlogLi_gplvm(pp,hyp,palpha,nf,Ycov,nsamp,KpriorInv,xgrid,Cnse,@covSEiso_curv,lambda);
% p0 = randn(size([params0(1:nx*nf)']));
% p0 = vec(fsamp-xgrid*alpha);
[f,df,kk,kk1] = lfunc(p0);
DerivCheck(lfunc,p0(:))
if mflag
    [params1, fval] = minFunc(lfunc,p0(:),options);
else
    params1 = fminunc(lfunc,p0(:),options);
end
[~,~,~,K_uu_anqiw_gp_rbf_nb] = lfunc(params1);
subplot(321); imagesc(K_uu_anqiw_gp_rbf_nb); colorbar, axis image, drawnow; title(['gp:' num2str(matrix_mse(Cmat-K_uu_anqiw_gp_rbf_nb))]);
subplot(322); plot([reshape(params1,[],nf)])
ff1 = reshape(params1,[],nf);

f1 = ff1;
rbf_p = hyp(1:2);
nsevar_p = exp(hyp(3));
C1 = covSEiso_curv([-rbf_p(1)/2;rbf_p(2)/2], f1+fmu);
subplot(221),imagesc(C1),colorbar,axis image,title(['gp:' num2str(matrix_mse(Cmat-C1))]);
subplot(222),imagesc(Cmat),axis image,colorbar
subplot(223),imagesc(corrcov(C1)),colorbar,axis image,title(['gp:' num2str(matrix_mse(corrcov(Cmat)-corrcov(C1)))]);
subplot(224),imagesc(corrcov(Cmat)),axis image,colorbar

% figure(3),clf
% hold on
% i1 = find(lbls(:,1)==1);
% xx = ff1(i1,1); yy = ff1(i1,2);
% scatter(xx,yy,'r')
% 
% i2 = find(lbls(:,2)==1);
% xx = ff1(i2,1); yy = ff1(i2,2);
% scatter(xx,yy,'b')
% 
% i3 = find(lbls(:,3)==1);
% xx = ff1(i3,1); yy = ff1(i3,2);
% scatter(xx,yy,'g')
% hold off
% 
% figure(1),clf,
subplot(121),imagesc(Cmat(1:20,1:20)),colorbar,axis image
subplot(122),imagesc(C1(1:20,1:20)),colorbar,axis image
    

%% u
figure(2)
ni = round(nx/5);
lfunc = @(pp) compNegEmbeddedGPlogLi_fgplvm(pp,hyp,palpha,ni,nf,Ycov,nsamp,KpriorInv,xgrid,Cnse,@covSEiso_curv,lambda);
pp0 = [p0; randn(nf*ni,1)];
[f,df] = lfunc(pp0);
% DerivCheck(lfunc,pp0(:))
if mflag
    [params1, fval] = minFunc(lfunc,pp0(:),options);
else
    params1 = fminunc(lfunc,pp0(:),options);
end
[~,~,K_uu_anqiw_gp_rbf_nb] = lfunc(params1);
subplot(323); imagesc(K_uu_anqiw_gp_rbf_nb); colorbar, axis image, drawnow; title(['gp:' num2str(matrix_mse(Cmat-K_uu_anqiw_gp_rbf_nb))]);
subplot(324); plot((reshape(params1(1:nx*nf),[],nf)))
ff2 = reshape(params1(1:nx*nf),[],nf);
uu2 = reshape(params1(1+nx*nf:nx*nf+ni*nf),[],nf);

f1 = ff2;
rbf_p = hyp(1:2);
nsevar_p = exp(hyp(3));
C1 = covSEiso_curv([-rbf_p(1)/2;rbf_p(2)/2], f1+fmu);
subplot(221),imagesc(C1),colorbar,axis image,title(['gp:' num2str(matrix_mse(Cmat-C1))]);
subplot(222),imagesc(Cmat),axis image,colorbar
subplot(223),imagesc(corrcov(C1)),colorbar,axis image,title(['gp:' num2str(matrix_mse(corrcov(Cmat)-corrcov(C1)))]);
subplot(224),imagesc(corrcov(Cmat)),axis image,colorbar

% figure(4),clf
% hold on
% i1 = find(lbls(:,1)==1);
% xx = ff2(i1,1); yy = ff2(i1,2);
% scatter(xx,yy,'r')
% 
% i2 = find(lbls(:,2)==1);
% xx = ff2(i2,1); yy = ff2(i2,2);
% scatter(xx,yy,'b')
% 
% i3 = find(lbls(:,3)==1);
% xx = ff2(i3,1); yy = ff2(i3,2);
% scatter(xx,yy,'g')
% hold off
% 
% figure(1),clf,
% subplot(121),imagesc(Cmat(1:20,1:20)),colorbar,axis image
% subplot(122),imagesc(C1(1:20,1:20)),colorbar,axis image
    
%% u noise
figure(2)
ni = round(nx/5);
lfunc = @(pp) compNegEmbeddedGPlogLi_fgplvm_noise(pp,hyp,palpha,ni,nf,Ycov,nsamp,KpriorInv,xgrid,Cnse,@covSEiso_curv,lambda);
pp0 = [p0; randn(nf*ni+nx,1)];
[f,df] = lfunc(pp0);
% DerivCheck(lfunc,pp0(:))
if mflag
    [params1, fval] = minFunc(lfunc,pp0(:),options);
else
    params1 = fminunc(lfunc,pp0(:),options);
end
[~,~,K_uu_anqiw_gp_rbf_nb] = lfunc(params1);
subplot(325); imagesc(K_uu_anqiw_gp_rbf_nb); colorbar, axis image, drawnow; title(['gp:' num2str(matrix_mse(Cmat-K_uu_anqiw_gp_rbf_nb))]);
ftrue = fsamp-xgrid*alpha;
subplot(326); plot([normalizecols(ftrue) normalizecols(reshape(params1(1:nx*nf),[],nf))])
ff3 = reshape(params1(1:nx*nf),[],nf);
uu3 = reshape(params1(1+nx*nf:nx*nf+ni*nf),[],nf);

figure(5),clf
subplot(211),plot_ff(fsamp-fmu,ff3,ff3,1,1)
subplot(212),plot_ff(fsamp-fmu,ff3,ff3,2,1)

