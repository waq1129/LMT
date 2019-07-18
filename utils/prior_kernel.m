function [BBwfun, BBwTfun, nu, sdiag, iikeep, Kprior] = prior_kernel(rhoxx,lenxx,nt,latentTYPE,tgrid)
if nargin<5
    tgrid = [1:nt]';
end
kSE = @(rho,len,x,z) covSEiso_deriv([log(len); log(rho)/2], x, z);
kAR1 = @(rho,len,x,z) covAR1iso([log(len); log(rho)/2], x, z);

switch latentTYPE
    case 1, % AR1
        %         covrow1 = rhoxx*exp(-(0:nt-1)/lenxx); % first row of covariance
        %         Kprior = toeplitz(covrow1); % Latent covariance
        %         [BBw, nu, sdiag, iikeep] = BfromK(Kprior+1e-4*eye(size(Kprior)));
        x = tgrid;
        xd = diff(x);
        avec = exp(-xd/lenxx); % for bidiagonal matrix A
        vvec = [rhoxx; rhoxx-rhoxx*avec.^2]; % for diagonal variance for each
        ss = 1./sqrt(vvec);
        ddB = ones(nt,1).*ss;
        ssB = -[avec;0].*circshift(ss,-1);
        nu = length(ss);
        sdiag = zeros(nu,1);
        iikeep = ones(nu,1);
        BBwfun = @(xx,invflag) BBwfun_AR(xx,ddB,ssB,invflag);
        BBwTfun = @(xx,invflag) BBwTfun_AR(xx,ddB,ssB,invflag);
        if nargout > 4
            B = spdiags([ssB, ddB], -1:0, nt,nt);
            BBw = pdinv(B);
            Kprior = BBw*BBw';
        end
    case 2, % SE
        Kprior = kSE(rhoxx,lenxx,tgrid,tgrid); % Latent covariance
        [BBw, nu, sdiag, iikeep] = BfromK(Kprior,1e-4);
        BBwfun = @(xx,invflag) BBwfun_SE(xx,BBw,invflag);
        BBwTfun = @(xx,invflag) BBwTfun_SE(xx,BBw,invflag);
end

function BBwxx = BBwfun_SE(xx,BBw,invflag)

if invflag
    BBwxx = BBw\xx;
else
    BBwxx = BBw*xx;
end

function BBwxx = BBwfun_AR(xx,ddB,ssB,invflag)

if invflag
    BBwxx = bidiagonal_low_multiply_matrix(ddB,ssB,xx);
else
    nt = length(xx);
    B = spdiags([ssB, ddB], -1:0, nt,nt);
    BBwxx = B\xx;
end

function BBwTxx = BBwTfun_SE(xx,BBw,invflag)

if invflag
    BBwTxx = BBw'\xx;
else
    BBwTxx = BBw'*xx;
end

function BBwTxx = BBwTfun_AR(xx,ddB,ssB,invflag)

if invflag
    BBwTxx = bidiagonal_up_multiply_matrix(ddB,ssB,xx);
else
    nt = length(xx);
    B = spdiags([ssB, ddB], -1:0, nt,nt);
    BBwTxx = B'\xx;
end






