function [K, blockK] = covSEiso(hyp, x, z, curvature, test)

% Squared Exponential covariance function with isotropic distance measure. The
% covariance function is parameterized as:
%
% k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
%
% where the P matrix is ell^2 times the unit matrix and sf^2 is the signal
% variance. The hyperparameters are:
%
% hyp = [ log(ell)
%         log(sf)  ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
%
% See also COVFUNCTIONS.M.
if nargin<2, K = '2'; return; end                  % report number of parameters
if nargin<3, z = []; end
if nargin<5, test = 0; end                                   % make sure, z exists
% make sure, z exists
if ~test, z = x; end
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

hyp(1) = max(hyp(1),-10);

ell = exp(hyp(1));                                 % characteristic length scale
sf2 = exp(2*hyp(2));                                           % signal variance
% precompute squared distances
if dg                                                               % vector kxx
    K = zeros(size(x,1),1);
else
    if xeqz                                                 % symmetric matrix Kxx
        K = sq_dist(x'/ell);
    else                                                   % cross covariances Kxz
        K = sq_dist(x'/ell,z'/ell);
    end
end
K = sf2*exp(-K/2);
if nargout >= 2
    if curvature==0
        blockK = K;
    end
    
    if curvature==1
        if test
            K = K';
            D = size(x,2);
            ds = cell(D,1);
            Ki = [];
            for ii=1:D
                xi = x(:,ii);
                zi = z(:,ii);
                di = bsxfun(@plus, zi(:), -xi(:)');
                ds{ii} = di;
                Ki = [Ki di.*K/ell^2];
            end
            
            blockK = [K Ki]';
        else
            if xeqz
                z = x;
            end
            D = size(x,2);
            ds = cell(D,1);
            Ki = [];
            for ii=1:D
                xi = x(:,ii);
                zi = z(:,ii);
                di = bsxfun(@plus, xi(:), -zi(:)');
                ds{ii} = di;
                Ki = [Ki di.*K/ell^2];
            end
            
            KK = [];
            for ii=1:D
                di = ds{ii};
                Kii = K.*(1-di.^2/ell^2)/ell^2;
                for jj=ii+1:D
                    dj = ds{jj};
                    Kii = [Kii -di.*dj/ell^4.*K];
                end
                Kii = [zeros(size(di,1), size(di,2)*(ii-1)) Kii];
                KK = [KK; Kii];
            end
            onesmask = kron(eye(D),ones(size(di)));
            KK1 = KK+KK';
            KK1(logical(onesmask)) = KK(logical(onesmask));
            
            blockK = [K Ki; Ki' KK1];
        end
    end
    
    if curvature==2
        if test
            K = K';
            D = size(x,2);
            ds = cell(D,1);
            K1 = [];
            for ii=1:D
                xi = x(:,ii);
                zi = z(:,ii);
                di = bsxfun(@plus, zi(:), -xi(:)');
                ds{ii} = di;
                K1 = [K1 di.*K/ell^2];
            end
            
            %%%%%%%%%%%%
            KK = [];
            for ii=1:D
                di = ds{ii};
                Kii = K.*(di.^2/ell^2-1)/ell^2;
                KK = [KK Kii];
            end
            K21 = KK;
            
            KK = [];
            for ii=1:D
                di = ds{ii};
                for jj=ii+1:D
                    dj = ds{jj};
                    KK = [KK K.*(di.*dj)/ell^4];
                end
            end
            K22 = KK;
            
            K2 = [K21 K22];
            
            blockK = [K K1 K2]';
            
        else
            if xeqz
                z = x;
            end
            
            nn = size(x,1);
            D = size(x,2);
            ds = cell(D,1);
            Ki = [];
            for ii=1:D
                xi = x(:,ii);
                zi = z(:,ii);
                di = bsxfun(@plus, xi(:), -zi(:)');
                ds{ii} = di;
                Ki = [Ki di.*K/ell^2];
            end
            
            KK = [];
            for ii=1:D
                di = ds{ii};
                Kii = K.*(1-di.^2/ell^2)/ell^2;
                for jj=ii+1:D
                    dj = ds{jj};
                    Kii = [Kii -di.*dj/ell^4.*K];
                end
                Kii = [zeros(size(di,1), size(di,2)*(ii-1)) Kii];
                KK = [KK; Kii];
            end
            
            onesmask = kron(eye(D),ones(size(di)));
            KK1 = KK+KK';
            KK1(logical(onesmask)) = KK(logical(onesmask));
            
            K1 = [K Ki; Ki' KK1];
            
            %%%%%%%%%%%%
            KK = [];
            KK1 = zeros(D*nn,D*nn);
            for ii=1:D
                di = ds{ii};
                Kii = K.*(di.^2/ell^2-1)/ell^2;
                KK = [KK Kii];
                KK1((ii-1)*nn+1:ii*nn,(ii-1)*nn+1:ii*nn) = K.*(3*di/ell^4-di.^3/ell^6);
                for jj=1:D
                    if jj==ii
                        continue;
                    end
                    dj = ds{jj};
                    KK1((ii-1)*nn+1:ii*nn,(jj-1)*nn+1:jj*nn) = K.*(di/ell^4-di.*dj.^2/ell^6);
                end
            end
            K21 = [KK; KK1];
            
            [r1, c1, pp] = gen_rc(D, eye(D));
            
            KK = [];
            for ii=1:D
                di = ds{ii};
                for jj=ii+1:D
                    dj = ds{jj};
                    KK = [KK K.*(di.*dj)/ell^4];
                end
            end
            
            KK1 = [];
            for ii=1:D
                di = ds{ii};
                KK11 = [];
                for jj=1:D*(D-1)/2
                    
                    if c1(jj)==ii
                        KK11 = [KK11 K.*(ds{r1(jj)}/ell^4-ds{r1(jj)}.*di.^2/ell^6)];
                    end
                    if r1(jj)==ii
                        KK11 = [KK11 K.*(ds{c1(jj)}/ell^4-ds{c1(jj)}.*di.^2/ell^6)];
                    end
                    if c1(jj)~=ii && r1(jj)~=ii
                        KK11 = [KK11 -K.*(di.*ds{c1(jj)}.*ds{r1(jj)})/ell^6];
                    end
                    
                end
                KK1 = [KK1; KK11];
            end
            K22 = [KK; KK1];
            
            K2 = [K21 K22];
            
            %%%%%%%%%%%%%
            KK1 = zeros(D*nn,D*nn);
            for ii=1:D
                di = ds{ii};
                KK1((ii-1)*nn+1:ii*nn,(ii-1)*nn+1:ii*nn) = K.*(3/ell^4-6*di.^2/ell^6+di.^4/ell^8);
                for jj=1:D
                    if jj==ii
                        continue;
                    end
                    dj = ds{jj};
                    KK1((ii-1)*nn+1:ii*nn,(jj-1)*nn+1:jj*nn) = K.*(1/ell^4-di.^2/ell^6-...
                        dj.^2/ell^6+di.^2.*dj.^2/ell^8);
                end
            end
            
            
            KK2 = [];
            for ii=1:D
                di = ds{ii};
                KK21 = [];
                for jj=1:D*(D-1)/2
                    
                    if c1(jj)==ii
                        KK21 = [KK21 -K.*(3*ds{r1(jj)}.*di/ell^6-ds{r1(jj)}.*di.^3/ell^8)];
                    end
                    if r1(jj)==ii
                        KK21 = [KK21 -K.*(3*ds{c1(jj)}.*di/ell^6-ds{c1(jj)}.*di.^3/ell^8)];
                    end
                    if c1(jj)~=ii && r1(jj)~=ii
                        KK21 = [KK21 -K.*(ds{c1(jj)}.*ds{r1(jj)}/ell^6-di.^2.*ds{c1(jj)}.*ds{r1(jj)}/ell^8)];
                    end
                    
                end
                KK2 = [KK2; KK21];
            end
            
            
            KK3 = [];
            for ii=1:D*(D-1)/2
                KK31 = [];
                for jj=1:D*(D-1)/2
                    ci = c1(ii);
                    ri = r1(ii);
                    cj = c1(jj);
                    rj = r1(jj);
                    
                    if ii==jj
                        KK31 = [KK31 K.*(1/ell^4-ds{ci}.^2/ell^6-ds{ri}.^2/ell^6+ds{ri}.^2.*ds{ci}.^2/ell^8)];
                    else
                        if ci==cj
                            KK31 = [KK31 -K.*(ds{ri}.*ds{rj}/ell^6-ds{ri}.*ds{ci}.*ds{rj}.*ds{cj}/ell^8)];
                        else
                            if ri==cj
                                KK31 = [KK31 -K.*(ds{ci}.*ds{rj}/ell^6-ds{ri}.*ds{ci}.*ds{rj}.*ds{cj}/ell^8)];
                            else
                                if ri==rj
                                    KK31 = [KK31 -K.*(ds{ci}.*ds{cj}/ell^6-ds{ri}.*ds{ci}.*ds{rj}.*ds{cj}/ell^8)];
                                else
                                    if ci==rj
                                        KK31 = [KK31 -K.*(ds{ri}.*ds{cj}/ell^6-ds{ri}.*ds{ci}.*ds{rj}.*ds{cj}/ell^8)];
                                    else
                                        KK31 = [KK31 K.*(ds{ri}.*ds{ci}.*ds{rj}.*ds{cj}/ell^8)];
                                    end
                                end
                            end
                        end
                        
                    end
                end
                KK3 = [KK3; KK31];
            end
            
            K3 = [KK1 KK2; KK2' KK3];
            
            blockK = [K1 K2; K2' K3];
        end
    end
end
end