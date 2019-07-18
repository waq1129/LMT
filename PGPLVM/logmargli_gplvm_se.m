function [L,m0,S0] = logmargli_gplvm_se(uu,BBwfun,ff,covfun,sigma2,nf,fflag)
if nargin<7
    fflag = 1;
end
if fflag
    % L = logmargli_gplvm(uu,BBwfun,yy,kerfun)
    %
    % Computes log marginal likelihood for GPLVM
    
    % Compute latent
    uu = reshape(uu,[],nf);
    xx = BBwfun(uu,0);
    
    C11 = covfun(xx,xx)+sigma2*eye(length(xx));
    
    m0 = 0;
    S0 = C11;
    % L = sum(logmvnpdf_multiplesamples(ff,m0,S0))-.5*uu'*uu;
    
    % Log-determinant term
    logdettrm = -.5*logdet(2*pi*S0);
    
    % Quadratic term
    Xctr = bsxfun(@minus,ff,m0);  % centered X
    Qtrm = -.5*sum(Xctr.*(S0\Xctr),1)';
    L = sum(Qtrm+logdettrm)-.5*trace(uu'*uu);
    L = -L;
else
    % L = logmargli_gplvm(uu,BBwfun,yy,kerfun)
    %
    % Computes log marginal likelihood for GPLVM
    
    % Compute latent
    uu = reshape(uu,[],nf);
    xx = BBwfun(uu,0);
    
    I = sigma2*eye(size(xx,1));
    C1 = covfun(xx(:,1),xx(:,1));
    C2 = covfun(xx(:,2),xx(:,2));
    
    % C11inv = inv(C1)+inv(C2)+sigma2*eye(size(xx,1));
    % C11 = covfun(xx,xx)+sigma2*eye(length(xx));
    
    m0 = 0;
    % S0 = C11;
    % L = sum(logmvnpdf_multiplesamples(ff,m0,S0))-.5*uu'*uu;
    
    % Log-determinant term
    logdettrm = .5*logdet(2*pi*(C1+C2+I*2))-.5*logdet(2*pi*(C1+I))-.5*logdet(2*pi*(C2+I));
    
    % Quadratic term
    Xctr = bsxfun(@minus,ff,m0);  % centered X
    Qtrm = -.5*sum(Xctr.*((C1+I)\Xctr+(C2+I)\Xctr),1)';
    L = sum(Qtrm+logdettrm)-.5*trace(uu'*uu);
    L = -L;
end